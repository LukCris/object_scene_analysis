# -*- coding: utf-8 -*-
"""
UI Gradio per la Scene Analysis:
- Tab "Singola scena": carica immagine, lancia analyze_scene e mostra overlay, tabelle, crops.
- Tab "Database Query": query immagine (sempre) o testo (solo CLIP).
"""

import time
from pathlib import Path
from typing import List

import gradio as gr
import pandas as pd
from PIL import Image

# Pipeline
from src.scene.analyze_scene import analyze_scene  # ritorna (eval_json, overlay_json)
from src.search.segment_matcher import load_index, topk_cosine, load_backend, embed_pil_generic

# ---------------------------
# Preset modello -> config
# ---------------------------

# Mappa dei preset esposti in UI: permette uno switch "one-click" tra CLIP (FT) e DINOv2.
# Ogni preset definisce i campi tecnici e abilita/disabilita la query testuale.
MODEL_PRESETS = {
    "CLIP ViT-B-32 (FT)": {
        "backend": "clip",
        "index_dir": "index_ft",
        "model_name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "finetuned": "runs/clip_ft_v1/best.pt",
        "dino_model": "",
        "out_dir": "outputs_scenes_ft",
        "text_query_enabled": True,
    },
    "DINOv2 ViT-S/14": {
        "backend": "dino",
        "index_dir": "index_dino",
        "model_name": "",                   # ignorato per DINOv2
        "pretrained": "",                   # ignorato per DINOv2
        "finetuned": "",                    # non usato
        "dino_model": "dinov2_vits14",
        "out_dir": "outputs_scenes_dino",
        "text_query_enabled": False,        # DINOv2 non supporta query testuale
    },
}

# ---------------------------
# Helpers
# ---------------------------

def _save_uploaded_image(img: Image.Image, upload_dir: Path) -> Path:
    """Salva l'immagine caricata dall'utente sotto `upload_dir` con timestamp.

    - Converte in RGB per robustezza (GIF/PNG con alpha, ecc.)
    - Usa qualità JPEG 95 (buon compromesso)
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = upload_dir / f"ui_scene_{ts}.jpg"
    img.convert("RGB").save(out_path.as_posix(), quality=95)
    return out_path

def _preds_to_df(preds: List[dict]) -> pd.DataFrame:
    """Converte la lista di predizioni (dict) in un DataFrame tabellare.

    Colonne: x1,y1,x2,y2,label,score — comodo per Dataframe Gradio.
    """
    rows = []
    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        rows.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "label": p["label"], "score": float(p.get("score", 0.0))
        })
    return pd.DataFrame(rows)

def _read_overlay_image(out_dir: Path, scene_path: Path) -> Image.Image | None:
    """Carica l'overlay renderizzato da analyze_scene."""
    over = out_dir / "overlay" / f"{scene_path.stem}_overlay.jpg"
    return Image.open(over.as_posix()).convert("RGB") if over.exists() else None

def _collect_crops(out_dir: Path, scene_path: Path) -> List[Image.Image]:
    """Raccoglie tutti i crops salvati per questa scena (ordinati per nome)."""
    crops_dir = out_dir / f"{scene_path.stem}_crops"
    if not crops_dir.exists():
        return []
    imgs = []
    for p in sorted(crops_dir.glob("*.jpg")):
        try:
            imgs.append(Image.open(p.as_posix()).convert("RGB"))
        except Exception:
            # salta eventuali file corrotti/non leggibili
            pass
    return imgs

# ---------------------------
# Callbacks
# ---------------------------

def on_model_change(model_choice: str):
    """
    Aggiorna i campi tecnici in base al preset modello scelto.
    Ritorna i nuovi valori per:
    index_dir, finetuned, model_name, pretrained, backend, dino_model, out_dir, e
    un update per rendere interattiva o meno la query testuale nel tab DB.
    """
    cfg = MODEL_PRESETS.get(model_choice, MODEL_PRESETS["CLIP ViT-B-32 (FT)"])
    tip = "Abilitata (CLIP)" if cfg["text_query_enabled"] else "Disabilitata per DINOv2"
    return (
        cfg["index_dir"],
        cfg["finetuned"],
        cfg["model_name"],
        cfg["pretrained"],
        cfg["backend"],
        cfg["dino_model"],
        cfg["out_dir"],
        tip,
        gr.update(interactive=cfg["text_query_enabled"])  # abilita/disabilita query testuale
    )

def _toggle_keep_topk(enabled: bool):
    """Attiva/disattiva l'interattività dello slider Top-K in base al checkbox."""
    return gr.update(interactive=bool(enabled))


def ui_analyze_single(
    image: Image.Image,
    index_dir: str,
    out_dir: str,
    finetuned: str,
    sam_ckpt: str,
    model_name: str,
    pretrained: str,
    backend: str,
    dino_model: str,
    enable_keep_topk: bool,
    keep_topk: int | None,
    min_area_px: int,
    max_masks: int,
    expand_bbox_ratio: float,
    bg_value: int,
    min_area_ratio: float,
    max_area_ratio: float,
    prefer_diff_label: bool,
    progress=gr.Progress(),
    db_backend: str="auto",
):
    """Callback eseguita al click di "Analizza scena".


    - Valida e normalizza input UI
    - Salva l'immagine in una cartella temporanea dell'output
    - Invoca `analyze_scene` con i parametri correnti
    - Prepara gli output per l'UI: overlay, tabelle, crops, link ai JSON
    """

    if image is None:
        raise gr.Error("Carica un'immagine prima di lanciare l'analisi.")

    # Gestione Top-K: se il toggle è OFF, non limitare (None)
    if enable_keep_topk:
        keep_topk = int(keep_topk) if keep_topk is not None else None
    else:
        keep_topk = None

    # Normalizzazione tipi scalari
    min_area_px = int(min_area_px)
    max_masks = int(max_masks) if max_masks else None
    bg_value = int(bg_value)
    min_area_ratio = float(min_area_ratio)
    max_area_ratio = float(max_area_ratio)
    expand_bbox_ratio = float(expand_bbox_ratio)

    # Paths e checkpoint
    index_dir = Path(index_dir)
    out_dir = Path(out_dir)
    finetuned = finetuned or None

    # 1) Salva l'immagine caricata
    scene_path = _save_uploaded_image(image, out_dir / "ui_uploads")

    # 2) Controlli rapidi (fail-fast con messaggio chiaro)
    if not (index_dir / "embeddings.npy").exists():
        raise gr.Error(f"Indice non trovato in: {index_dir}/embeddings.npy")
    if not Path(sam_ckpt).exists():
        raise gr.Error(f"Checkpoint SAM non trovato: {sam_ckpt}")

    progress(0.2, desc="Analisi scena in corso…")

    # 3) Esegui pipeline core
    # Nota: per backend='dino', model_name/pretrained (CLIP) vengono ignorati internamente.
    eval_json, overlay_json = analyze_scene(
        scene_path=scene_path,
        index_dir=index_dir,
        out_dir=out_dir,
        sam_ckpt=sam_ckpt,
        clip_model=model_name,
        pretrained=pretrained,
        backend=backend,
        dino_model=dino_model,
        min_area=min_area_px,
        topk=5,
        max_masks=max_masks,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        keep_topk_objects=keep_topk,
        expand_bbox_ratio=expand_bbox_ratio,
        bg_value=bg_value,
        finetuned=finetuned if backend == "clip" else None,
        search_backend=db_backend,
        prefer_diff_label=bool(prefer_diff_label),
    )

    progress(0.7, desc="Preparazione output…")

    # 4) Prepara output UI: immagine overlay, tabelle e galleria crops
    overlay_img = _read_overlay_image(out_dir, scene_path)
    df_eval = _preds_to_df(eval_json["preds"])
    df_overlay = _preds_to_df(overlay_json["preds"])
    crops = _collect_crops(out_dir, scene_path)

    # Link ai JSON su disco per download
    json_dir = out_dir / "json"
    eval_file = json_dir / f"{scene_path.stem}.json"
    overlay_file = json_dir / f"{scene_path.stem}_overlay.json"

    progress(1.0)
    return overlay_img, df_eval, df_overlay, crops, str(eval_file), str(overlay_file)

def ui_db_query(
    model_choice: str,
    q_img, q_txt,
    index_dir, finetuned, model_name, pretrained, backend, dino_model, k=8
):
    """Esegue una query nel DB vettoriale.


    - Con immagine (CLIP/DINO): embed della query con la stessa pipeline dell'indice
    - Con testo (solo CLIP): encode_text + L2-normalize
    - Top-K naive in RAM (per semplicità UI): usa `topk_cosine`
    """
    index_dir = Path(index_dir)
    if not (index_dir / "embeddings.npy").exists():
        raise gr.Error(f"Indice non trovato: {index_dir}/embeddings.npy")

    # Carica backend (clip o dino) in base al preset
    backend_tuple = load_backend(
        backend=backend,
        clip_model= model_name or "ViT-B-32",
        pretrained=pretrained or "laion2b_s34b_b79k",
        dino_model=dino_model or "dinov2_vits14",
        finetuned=(finetuned or None),
    )
    mat, meta = load_index(index_dir)

    # Embedding della query
    if q_img is not None:
        # Query immagine → embedding con lo stesso backend dell'indice
        z = embed_pil_generic(q_img.convert("RGB"), backend_tuple)
    else:
        # Query testuale: solo se backend=CLIP
        if backend != "clip":
            raise gr.Error("La query testuale è disponibile solo con CLIP.")
        if not q_txt or not q_txt.strip():
            raise gr.Error("Fornisci un'immagine oppure un testo di query.")
        import torch, open_clip
        _, model, _, device = backend_tuple  # ("clip", model, preprocess, device)
        tok = open_clip.get_tokenizer(model_name or "ViT-B-32")([q_txt.strip()]).to(device)
        with torch.no_grad():
            zt = model.encode_text(tok)
            import torch.nn.functional as F
            zt = F.normalize(zt, dim=-1)
        z = zt.cpu()

    # Top-K
    vals, idxs = topk_cosine(z, mat, k=int(k))
    rows, thumbs = [], []
    for s, j in zip(vals, idxs):
        m = meta[int(j)]
        p = m["path"]; lab = m["label"]
        rows.append({"label": lab, "score": float(s), "path": p})
        try:
            thumbs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass

    df = pd.DataFrame(rows, columns=["label","score","path"])
    return thumbs, df

# ---------------------------
# Gradio UI
# ---------------------------

def build_ui():
    """Costruisce e avvia l'app Gradio con due tab: Singola scena e Database Query."""
    with gr.Blocks(title="Naruto Scene Analysis", theme="soft") as demo:
        gr.Markdown("## Scene Analysis — SAM + CLIP/DINOv2")
        gr.Markdown("Modelli supportati: **CLIP ViT-B-32 (fine-tuned)** e **DINOv2 ViT-S/14**.")
        gr.Markdown("Personaggi riconoscibili: **Naruto**, **Gaara**, **Tsunade**, **Sakura**.")

        with gr.Tabs():
            # ------------------- Singola Scena -------------------
            with gr.Tab("Singola scena"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Scelta modello (popola i campi tecnici sottostanti)
                        model_choice = gr.Dropdown(
                            choices=list(MODEL_PRESETS.keys()),
                            value="CLIP ViT-B-32 (FT)",
                            label="Embedding model"
                        )

                        image = gr.Image(type="pil", label="Immagine scena")

                        # Campi tecnici: auto-popolati e non interattivi
                        index_dir = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["index_dir"],
                                               label="Indice (cartella)", interactive=False)
                        finetuned = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["finetuned"],
                                               label="Checkpoint CLIP FT (.pt)", interactive=False)
                        sam_ckpt = gr.Textbox(value="weights/sam_vit_b_01ec64.pth",
                                              label="Checkpoint SAM (.pth)")
                        model_name = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["model_name"],
                                                label="CLIP model", interactive=False)
                        pretrained = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["pretrained"],
                                                label="CLIP pretrained tag", interactive=False)
                        backend = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["backend"],
                                             label="Backend", interactive=False)
                        db_backend = gr.Dropdown(choices=["auto", "naive", "faiss"], value="auto", label="DB backend")
                        dino_model = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["dino_model"],
                                                label="DINOv2 model", interactive=False)
                        out_dir = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["out_dir"],
                                             label="Output dir")

                        # Parametri leggeri
                        gr.Markdown("### Parametri (leggeri)")
                        enable_keep_topk = gr.Checkbox(
                            value=False, label="Abilita Top-K overlay",
                            info="Se attivo, mostra solo i K oggetti migliori nell'overlay"
                        )
                        keep_topk = gr.Slider(
                            1, 5, value=1, step=1, label="Top-K overlay (solo UI)",
                            info="Seleziona per visualizzare solo i migliori K overlay",
                            interactive=False  # disabilitato finché il toggle è OFF
                        )
                        prefer_diff_label = gr.Checkbox(
                            value=False,
                            label="Preferisci etichette diverse nei Top-K",
                            info="Quando selezioni più oggetti (Top-K), prova a scegliere box con label diverse se i punteggi sono vicini"
                        )
                        min_area_px = gr.Number(value=10000, label="SAM min area (px)", precision=0)
                        max_masks = gr.Slider(1, 16, value=12, step=1, label="SAM max masks")
                        expand_bbox_ratio = gr.Slider(0.0, 0.3, value=0.18, step=0.01, label="Espansione bbox (ratio)")
                        bg_value = gr.Dropdown(choices=[0,128,220,255], value=0, label="BG compositing (RGBA→RGB)")
                        min_area_ratio = gr.Slider(0.0, 0.2, value=0.04, step=0.01, label="Area min (ratio immagine)")
                        max_area_ratio = gr.Slider(0.2, 1.0, value=0.55, step=0.01, label="Area max (ratio immagine)")

                        run_btn = gr.Button("Analizza scena", variant="primary")

                    with gr.Column(scale=1):
                        # Output UI
                        overlay_img = gr.Image(label="Overlay", interactive=False)
                        df_eval = gr.Dataframe(headers=["x1","y1","x2","y2","label","score"], label="Predizioni (EVAL JSON)")
                        df_overlay = gr.Dataframe(headers=["x1","y1","x2","y2","label","score"], label="Predizioni (OVERLAY JSON)")
                        crops_gallery = gr.Gallery(label="Crops", columns=4, height=240)
                        eval_json_file = gr.File(label="Scarica JSON (eval)")
                        overlay_json_file = gr.File(label="Scarica JSON (overlay)")

                # Quando cambia il preset, aggiorna i campi tecnici e lo stato della query testuale nel tab DB
                model_choice.change(
                    fn=on_model_change,
                    inputs=[model_choice],
                    outputs=[index_dir, finetuned, model_name, pretrained, backend, dino_model, out_dir,
                             gr.Textbox(label="(solo info) Stato query testuale"),  # placeholder invisibile
                             ],
                )

                # Attiva/disattiva lo slider Top-K in base al checkbox
                enable_keep_topk.change(
                    fn=_toggle_keep_topk,
                    inputs=[enable_keep_topk],
                    outputs=[keep_topk],
                )

                run_btn.click(
                    ui_analyze_single,
                    inputs=[
                        image,
                        index_dir, out_dir, finetuned, sam_ckpt,
                        model_name, pretrained, backend, dino_model,
                        enable_keep_topk,
                        keep_topk,
                        min_area_px, max_masks,
                        expand_bbox_ratio, bg_value,
                        min_area_ratio, max_area_ratio,
                        prefer_diff_label,
                        db_backend,
                    ],
                    outputs=[overlay_img, df_eval, df_overlay, crops_gallery, eval_json_file, overlay_json_file]
                )

            # ------------------- Database Query -------------------
            with gr.Tab("Database Query"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice_q = gr.Dropdown(
                            choices=list(MODEL_PRESETS.keys()),
                            value="CLIP ViT-B-32 (FT)",
                            label="Embedding model (per Query)"
                        )
                        q_img = gr.Image(type="pil", label="Query immagine (opzionale)")
                        q_txt = gr.Textbox(label="Query testuale (solo CLIP)")
                        # campi tecnici auto-popolati
                        index_dir_q  = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["index_dir"],
                                                  label="Indice (cartella)", interactive=False)
                        finetuned_q  = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["finetuned"],
                                                  label="Checkpoint CLIP FT (.pt)", interactive=False)
                        model_name_q = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["model_name"],
                                                  label="CLIP model", interactive=False)
                        pretrained_q = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["pretrained"],
                                                  label="CLIP pretrained tag", interactive=False)
                        backend_q    = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["backend"],
                                                  label="Backend", interactive=False)
                        dino_model_q = gr.Textbox(value=MODEL_PRESETS["CLIP ViT-B-32 (FT)"]["dino_model"],
                                                  label="DINOv2 model", interactive=False)
                        topk_q = gr.Slider(1, 20, value=8, step=1, label="Top-K")
                        run_q = gr.Button("Cerca", variant="primary")

                    with gr.Column(scale=1):
                        gallery_q = gr.Gallery(label="Risultati", columns=4, height=320)
                        df_q = gr.Dataframe(headers=["label", "score", "path"], label="Match")

                # Cambiando il preset, sincronizza i campi tecnici e abilita/disabilita la textbox testuale
                def _on_model_change_db(choice):
                    cfg = MODEL_PRESETS.get(choice, MODEL_PRESETS["CLIP ViT-B-32 (FT)"])
                    return (
                        cfg["index_dir"], cfg["finetuned"], cfg["model_name"], cfg["pretrained"],
                        cfg["backend"], cfg["dino_model"],
                        gr.update(interactive=cfg["text_query_enabled"],
                                  placeholder="Testo (solo CLIP)" if cfg["text_query_enabled"] else "Disabilitato per DINOv2")
                    )

                model_choice_q.change(
                    fn=_on_model_change_db,
                    inputs=[model_choice_q],
                    outputs=[index_dir_q, finetuned_q, model_name_q, pretrained_q, backend_q, dino_model_q, q_txt]
                )

                run_q.click(
                    ui_db_query,
                    inputs=[model_choice_q, q_img, q_txt,
                            index_dir_q, finetuned_q, model_name_q, pretrained_q, backend_q, dino_model_q, topk_q],
                    outputs=[gallery_q, df_q]
                )
        # Avvio app: coda abilitata per gestire più richieste, server locale, no share pubblico
        demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=7860, show_error=True, share=False)
    return demo


if __name__ == "__main__":
    build_ui()
