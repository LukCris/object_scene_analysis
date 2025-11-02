"""
analyze_scene.py

Scopo
-----
Data una scena (immagine), esegue:
1) Segmentazione automatica con SAM -> maschere candidate
2) Post-processing maschere -> crop RGBA con contesto, conversione a PIL RGB
3) Embedding (CLIP o DINOv2) + ricerca naive (coseno) o FAISS (IP)
4) Assegnazione label/top-1 per ciascun segmento
5) Salvataggio risultati:
   - `json/<scene>.json`           (tutte le predizioni, nessuna soglia/NMS)
   - `json/<scene>_overlay.json`   (predizioni filtrate/NMS per visualizzazione)
   - `overlay/<scene>_overlay.jpg` (overlay disegnato)
   - `<scene>_crops/*`             (crop per ogni segmento)

Note
----
- È presente TTA di embedding (flip orizzontale) per maggiore stabilità.
- Filtri geometrici (`min/max_area_ratio`) e NMS leggero ripuliscono l'overlay.
"""

import argparse, json, cv2

import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.sam_infer.segment_scene import (
    generate_masks,
    load_sam_automatic,
    mask_to_bbox,
)
from src.scene.segment_processor import (
    apply_mask_rgba,
    crop_mask_tight,
    rgba_to_rgb_for_clip,
)
from src.search.faiss_db import faiss_exists
from src.search.segment_matcher import (
    decide_label_baseline,
    decide_label_faiss,
    embed_pil_generic,
    load_backend,
    load_index,
    topk_cosine,
)


# -----------------------------------------------------------------------------
# Visualizzazione overlay
# -----------------------------------------------------------------------------

def draw_overlay(
        image_rgb: np.ndarray,
        boxes_labels_scores,
        out_path: Path
) -> None:
    """Disegna box, riempimento semi-trasparente e cartellini label/score.

    `boxes_labels_scores` è una lista di tuple `(x1, y1, x2, y2, label, score)`.
    """
    CLASS_COLORS = {
        "Naruto": (66, 133, 244),  # blu
        "Gara": (234, 67, 53),  # rosso
        "Sakura": (219, 39, 119),  # magenta
        "Tsunade": (52, 168, 83),  # verde
    }
    DEFAULT_COLOR = (255, 140, 0)  # arancione fallback

    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2, lbl, sc) in boxes_labels_scores:
        color = CLASS_COLORS.get(lbl, DEFAULT_COLOR)
        # riempimento semi-trasparente
        drw.rectangle([(x1, y1), (x2, y2)], fill=color + (64,))
        # bordo pieno
        drw.rectangle([(x1, y1), (x2, y2)], outline=color + (255,), width=2)

        text = f"{lbl} ({sc:.2f})"
        try:
            bbox = drw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:  # compat vecchie PIL
            tw, th = drw.textsize(text, font=font)

        pad = 2
        tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
        tx2, ty2 = x1 + tw + 2 * pad, y1
        drw.rectangle(
            [(tx1, ty1), (tx2, ty2)],
            fill=(0, 0, 0, 200),
            outline=color + (255,),
            width=1,
        )
        drw.text(
            (tx1 + pad, ty1 + pad),
            text,
            fill=(255, 255, 255, 255),
            font=font
        )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path.as_posix())


# -----------------------------------------------------------------------------
# NMS e selezione top-K distinti
# -----------------------------------------------------------------------------

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    iw = max(0, x2 - x1 + 1)
    ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    aA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    aB = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    uni = aA + aB - inter
    return inter / uni if uni > 0 else 0.0

def _nms_preds(boxes_labels_scores, iou_thr: float = 0.30):
    """NMS (Non-Maximum Suppression) leggero sulle predizioni overlay basato su IoU.

    Mantiene le box in ordine di score decrescente ed elimina quelle con
    IoU >= `iou_thr` (si sovrappone troppo) con una box già tenuta.

    Effetto: se SAM produce più segmenti/bbox quasi identici sullo stesso personaggio,
    ne mostra uno solo (il migliore).
    """
    keep, used = [], [False] * len(boxes_labels_scores)

    order = sorted(
        range(len(boxes_labels_scores)), key=lambda i: boxes_labels_scores[i][5], reverse=True
    )
    for i in order:
        if used[i]:
            continue
        used[i] = True
        keep.append(boxes_labels_scores[i])
        for j in order:
            if used[j]:
                continue
            if iou(boxes_labels_scores[i], boxes_labels_scores[j]) >= iou_thr:
                used[j] = True
    return keep

def _select_topK_distinct(
    bls,
    K: int = 5,
    score_backoff=(0.80, 0.70, 0.60),
    iou_thr: float = 0.45,
    prefer_diff_label: bool = False,
    delta: float = 0.05,
):
    """Selezione euristica di K box distinte e con score elevato.

    - `score_backoff`: tentativi a soglie decrescenti (es. 0.80 -> 0.70 -> 0.60)
    - `prefer_diff_label`: se True, quando ha già preso una sola box e la seconda
       migliore ha stessa label, prova a sostituirla con una di label diversa se:
       non si sovrappone troppo agli altri pick, e lo score è comparabile (delta=0.05) oppure supera la soglia corrente.
    """

    # ordino per score decrescente
    bls = sorted(bls, key=lambda t: t[5], reverse=True)
    for thr in score_backoff:
        # filtro i candidati con score >= thr
        cands = [b for b in bls if b[5] >= thr]
        picked = []
        for cand in cands:
            # conservo solo quelli con IoU < iou_thr (che non si sovrappongono molto))
            if any(iou(cand, p) >= iou_thr for p in picked):
                continue
            # seleziono, se richiesto, label diverse
            if prefer_diff_label and len(picked) == 1 and cand[4] == picked[0][4]:
                alt = next(
                    (
                        b
                        for b in cands
                        if b[4] != picked[0][4] and all(iou(b, p) < iou_thr for p in picked)
                    ),
                    None,
                )
                if alt and (picked[0][5] - alt[5] <= delta or alt[5] >= thr):
                    cand = alt
            picked.append(cand)
            # se raggiungo il massimo mi fermo
            if len(picked) == K:
                return picked

    picked = []
    for cand in bls:
        if all(iou(cand, p) < iou_thr for p in picked):
            picked.append(cand)
            if len(picked) == K:
                break
    return picked


# -----------------------------------------------------------------------------
# Helpers TTA e bbox
# -----------------------------------------------------------------------------

def _expand_bbox(x1, y1, x2, y2, H, W, ratio: float = 0.08):
    """Espande il bbox di una piccola frazione (default 8%) per più contesto.

    Utile a migliorare l'IoU con GT e a includere dettagli limitrofi.
    """
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    dx = int(w * ratio / 2.0)
    dy = int(h * ratio / 2.0)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(W - 1, x2 + dx)
    ny2 = min(H - 1, y2 + dy)
    return nx1, ny1, nx2, ny2


def embed_tta(pil_rgb: Image.Image, backend_tuple):
    """Test-time augmentation: media embedding dell'immagine e del suo flip orizzontale."""
    z1 = embed_pil_generic(pil_rgb, backend_tuple)
    z2 = embed_pil_generic(pil_rgb.transpose(Image.FLIP_LEFT_RIGHT), backend_tuple)
    return (z1 + z2) / 2.0


# -----------------------------------------------------------------------------
# Pipeline principale
# -----------------------------------------------------------------------------

def analyze_scene(
    scene_path: Path,
    index_dir: Path = Path("index"),
    out_dir: Path = Path("outputs"),
    # ricerca
    search_backend: str = "auto",  # "auto" -> usa FAISS se presente, altrimenti naive
    # modelli
    sam_ckpt: str = "weights/sam_vit_b_01ec64.pth",
    pretrained: str = "laion2b_s34b_b79k",
    clip_model: str = "ViT-B-32",
    finetuned: str | None = None,
    dino_model: str = "dinov2_vits14",
    backend: str | None = None,  # {"clip","dino"}
    # filtri/parametri scena
    min_area: int = 15000,
    topk: int = 5,
    max_masks: int | None = 8,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.60,
    keep_topk_objects: int | None = None,
    expand_bbox_ratio: float = 0.08,
    bg_value: int = 128,
    prefer_diff_label: bool = False,
):
    """Esegue l'intera pipeline di analisi di una scena.

    Returns
    -------
    (eval_json, overlay_json)
        Dizionari con le predizioni per valutazione e per overlay.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    # Carica immagine
    bgr = cv2.imread(scene_path.as_posix())
    if bgr is None:
        raise FileNotFoundError(scene_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # Backend di embedding (CLIP/DINO) e indice (naive)
    backend_tuple = load_backend(
        backend=backend,
        clip_model=clip_model,
        pretrained=pretrained,
        dino_model=dino_model,
        finetuned=finetuned,
    )
    mat, meta = load_index(index_dir)

    # Attiva FAISS se richiesto/disponibile
    use_faiss = (search_backend in ("auto", "faiss")) and faiss_exists(index_dir)

    # Directory per salvare i crop
    crops_dir = out_dir / (scene_path.stem + "_crops")
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Segmentazione SAM
    sam_gen = load_sam_automatic(ckpt=sam_ckpt, model_type="vit_b")
    masks = generate_masks(bgr, sam_gen, min_area=min_area, max_masks=max_masks)

    preds_all = []  # per valutazione (no soglia/NMS)
    overlay_cands = []  # per overlay (con soglia/NMS)

    for i, m in enumerate(masks, start=1):
        seg = m["segmentation"].astype(bool)
        x1, y1, x2, y2 = mask_to_bbox(seg)
        if x2 <= x1 or y2 <= y1:
            continue

        # Filtri geometrici leggeri
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        ratio = area / float(H * W)
        if ratio < float(min_area_ratio) or ratio > float(max_area_ratio):
            continue

        # Espansione bbox per più contesto
        x1e, y1e, x2e, y2e = _expand_bbox(x1, y1, x2, y2, H, W, ratio=expand_bbox_ratio)

        # Cutout + crop "tight" con pad relativo
        rgba = apply_mask_rgba(bgr, seg)
        bw, bh = x2e - x1e + 1, y2e - y1e + 1
        rel_pad = max(8, int(0.10 * min(bw, bh)))
        crop_rgba = crop_mask_tight(rgba, seg, pad=rel_pad)
        pil_rgb = rgba_to_rgb_for_clip(crop_rgba, bg_value=bg_value)

        # Embedding con TTA (flip orizzontale)
        z = embed_tta(pil_rgb, backend_tuple)

        # Matching (FAISS se disponibile, altrimenti naive)
        if use_faiss:
            label, score, _dbg = decide_label_faiss(z, index_dir, meta)
        else:
            label, score, _dbg = decide_label_baseline(z, mat, meta)

        # EVAL: nessuna soglia, nessun NMS
        preds_all.append((x1e, y1e, x2e, y2e, label, float(score)))

        # OVERLAY: salviamo comunque il crop e candidiamo la box
        crop_path = crops_dir / f"seg_{i:02d}_{label}_{score:.2f}.jpg"
        pil_rgb.save(crop_path.as_posix())
        overlay_cands.append((x1e, y1e, x2e, y2e, label, float(score)))

    # Overlay: NMS leggero + top-K opzionale
    overlay_nms = _nms_preds(overlay_cands, iou_thr=0.30)
    overlay_final = overlay_nms
    if keep_topk_objects is not None:
        overlay_final = _select_topK_distinct(
            overlay_nms,
            K=int(keep_topk_objects),
            score_backoff=(0.80, 0.70, 0.60),
            iou_thr=0.50,
            prefer_diff_label=prefer_diff_label,
            delta=0.10,
        )

    # JSON EVAL (ufficiale)
    eval_json = {
        "scene": scene_path.name,
        "preds": [
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": lbl,
                "score": float(sc),
            }
            for (x1, y1, x2, y2, lbl, sc) in preds_all
        ],
    }
    (out_dir / f"json/{scene_path.stem}.json").write_text(
        json.dumps(eval_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # JSON OVERLAY (filtrato)
    overlay_json = {
        "scene": scene_path.name,
        "preds": [
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": lbl,
                "score": float(sc),
            }
            for (x1, y1, x2, y2, lbl, sc) in overlay_final
        ],
    }
    (out_dir / f"json/{scene_path.stem}_overlay.json").write_text(
        json.dumps(overlay_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Overlay disegnato
    overlay_path = out_dir / f"overlay/{scene_path.stem}_overlay.jpg"
    draw_overlay(rgb, overlay_final, overlay_path)

    # Log sintetico su console
    print(
        f"\nScene: {scene_path.name}  |  eval preds: {len(preds_all)}  |  overlay: {len(overlay_final)}"
    )
    for (x1, y1, x2, y2, lbl, sc) in overlay_final:
        print(f"[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] -> {lbl:12s}  score={sc:.3f}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Crops dir   : {crops_dir}")

    return eval_json, overlay_json


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--search-backend", default="auto", choices=["auto", "naive", "faiss"])
    ap.add_argument("--index", default="index")
    ap.add_argument("--out", default="outputs")

    ap.add_argument("--sam-ckpt", default="weights/sam_vit_b_01ec64.pth")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")

    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument(
        "--finetuned", default=None, help="Percorso al file best.pt prodotto con il fine-tuning"
    )
    ap.add_argument("--dino-model", default="dinov2_vits14")
    ap.add_argument("--backend", choices=["clip", "dino"], default="clip")

    ap.add_argument("--min-area", default=10000, type=int)
    ap.add_argument("--topk", default=5, type=int)
    ap.add_argument("--max-masks", default=12, type=int)
    ap.add_argument("--min-area-ratio", default=0.04, type=float)
    ap.add_argument("--max-area-ratio", default=0.55, type=float)
    ap.add_argument("--keep-topk-objects", type=int, default=None)
    ap.add_argument("--expand-bbox-ratio", type=float, default=0.18)
    ap.add_argument("--bg-value", type=int, default=0)
    ap.add_argument("--prefer-diff-label", action="store_true", default=False)

    args = ap.parse_args()

    analyze_scene(
        scene_path=Path(args.scene),
        index_dir=Path(args.index),
        out_dir=Path(args.out),
        sam_ckpt=args.sam_ckpt,
        clip_model=args.clip_model,
        pretrained=args.pretrained,
        min_area=args.min_area,
        topk=args.topk,
        max_masks=args.max_masks,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        keep_topk_objects=args.keep_topk_objects,
        expand_bbox_ratio=args.expand_bbox_ratio,
        bg_value=args.bg_value,
        finetuned=args.finetuned,
        backend=args.backend,
        dino_model=args.dino_model,
        search_backend=args.search_backend,
        prefer_diff_label=args.prefer_diff_label,
    )


if __name__ == "__main__":
    main()
