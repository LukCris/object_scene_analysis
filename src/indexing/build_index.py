"""
build_index.py

Obiettivo
---------
Creare l'indice vettoriale per la fase di retrieval. A partire da un `manifest`
(`CSV` con colonne `path,label`) calcola gli embedding immagine con CLIP oppure
con DINOv2, applica alcune scelte di qualità (cap bilanciato per classe,
deduplicazione intra-classe tramite similarità coseno, opzionale TTA flip), e
salva:

- `embeddings.npy`  : matrice (N, D) L2-normalizzata (coseno = prodotto interno)
- `meta.jsonl`      : JSON lines con `{path,label,aug}` per ogni riga di `embeddings`
- `stats.json`      : conteggi per classe e totale
- (opzionale) indice FAISS `flat/IP` sui vettori unitari

Nota: se viene passato un checkpoint `--finetuned` di CLIP, il codice lo carica
per gestire differenze di chiavi.
"""

import argparse, json, random, torch, open_clip

from pathlib import Path
from typing import Dict
from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as T

from src.search.faiss_db import build_faiss_index


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def gamma_pil(img: Image.Image, gamma: float) -> Image.Image:
    """Applica una correzione gamma a un'immagine PIL e restituisce una nuova PIL.

    Utile per data augmentation di luminanza (se attivata esplicitamente). Non è
    abilitata di default per non alterare la distribuzione dell'indice.
    """
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr ** gamma, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def load_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str | None = None,
    finetuned: str | None = None,
):
    """Carica CLIP (open_clip) + preprocess e opzionalmente pesi fine-tuned.

    Parameters
    ----------
    model_name : str
        Architettura CLIP (es. "ViT-B-32").
    pretrained : str
        Tag dei pesi pre-addestrati (open_clip).
    device : Optional[str]
        Dispositivo ('cuda' se disponibile, altrimenti 'cpu').
    finetuned : Optional[str]
        Path di un checkpoint fine-tuned (.pt).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()

    if finetuned:
        ckpt = torch.load(finetuned, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[finetuned] loaded: {finetuned}  miss={len(missing)}  unexp={len(unexpected)}"
        )

    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def load_dino(model_name: str = "dinov2_vits14", device: str | None = None):
    """Carica DINOv2 via torch.hub e definisce il preprocess standard 224x224.

    Nota: richiede la repo `facebookresearch/dinov2`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model = model.to(device).eval()

    preprocess = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return model, preprocess, device


def read_manifest_grouped(manifest_csv: Path) -> dict[str, list[str]]:
    """Legge un manifest `path,label` e raggruppa i path per etichetta.

    Returns
    -------
    dict[str, list[str]]
        Mappa `label -> [paths...]`.

    Raises
    ------
    ValueError
        Se l'header non è esattamente `path,label` (case-insensitive).
    """
    df = pd.read_csv(manifest_csv)
    cols = [c.strip().lower() for c in df.columns]
    if cols != ["path", "label"]:
        raise ValueError(
            f"Manifest {manifest_csv} deve avere header 'path,label', trovato: {df.columns.tolist()}"
        )
    groups: dict[str, list[str]] = {}
    for p, l in zip(df["path"].astype(str), df["label"].astype(str)):
        groups.setdefault(l, []).append(p)
    return groups


def batched(seq: list, batch_size: int):
    """Iteratore che restituisce chunk consecutivi di `seq` di dimensione `batch_size`."""
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------

def build_index(
    manifest_csv: Path,
    out_dir: Path,
    batch_size: int = 64,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    finetuned: str | None = None,
    aug_index: bool = False,
    backend: str = "clip",
    faiss: bool = False,
    dino_model: str = "dinov2_vits14",
    # --- parametri qualità ---
    max_per_class: int = 100,
    dedup_thr: float = 0.985,
    tta_flip: bool = True,
    seed: int = 42,
):
    """Costruisce un indice bilanciato e deduplicato da un manifest.

    Politiche implementate
    ----------------------
    - Cap per classe: `max_per_class` limita il numero di esempi per etichetta.
      Prima si campionano fino a `3x cap` per dare più materiale alla dedup.
    - Deduplicazione intra-classe: un nuovo embedding viene scartato se la
      massima similarità coseno con quelli già accettati della stessa classe è
      `>= dedup_thr`.
    - TTA flip: esegue anche il flip orizzontale e media i vettori (stabilizza).

    Persistenza
    -----------
    - `embeddings.npy` (N, D) vettori unitari (L2=1)
    - `meta.jsonl` per allineare ogni riga a `path,label,aug`
    - `stats.json` riepilogo per classe
    - (opz) indice FAISS `flat/IP` se `faiss=True`
    """
    # Semi per riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Selezione backend
    if backend == "clip":
        model, preprocess, _, device = load_clip(
            model_name, pretrained, device=None, finetuned=(finetuned or None)
        )

        @torch.no_grad()
        def _embed_one(pil_img: Image.Image) -> np.ndarray:
            x = preprocess(pil_img).unsqueeze(0).to(device)
            z = model.encode_image(x)
            z = F.normalize(z, dim=-1)
            if tta_flip:
                xf = preprocess(pil_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
                zf = model.encode_image(xf)
                zf = F.normalize(zf, dim=-1)
                z = F.normalize((z + zf) / 2.0, dim=-1)
            return z.cpu().numpy()  # (1, D)

        print(
            f"[index] backend=CLIP  model={model_name}  pretrained={pretrained}  "
            f"finetuned={'yes' if finetuned else 'no'}  tta_flip={tta_flip}"
        )

    elif backend == "dino":
        model, preprocess, device = load_dino(dino_model, device=None)

        @torch.no_grad()
        def _embed_one(pil_img: Image.Image) -> np.ndarray:
            x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
            z = model(x)
            if isinstance(z, (list, tuple)):
                z = z[0]
            z = F.normalize(z, dim=-1)
            return z.cpu().numpy()

        print(f"[index] backend=DINOv2  model={dino_model}  tta_flip=False")
    else:
        raise ValueError("Il backend deve essere 'clip' o 'dino'")

    # Carica manifest e ordina etichette per replicabilità dell'output
    groups = read_manifest_grouped(manifest_csv)
    labels = sorted(groups.keys())
    print(
        f"Manifest: {manifest_csv}  |  classi: {labels}  |  cap per classe: {max_per_class}"
    )

    # Buffer globali
    all_embs: list[np.ndarray] = []
    all_meta: list[dict] = []

    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()  # ricominciamo da zero

    with meta_path.open("w", encoding="utf-8") as fmeta:
        for lab in labels:
            paths = groups[lab]
            random.shuffle(paths)

            # Pre-campionamento generoso (fino a 3x cap) per aiutare la dedup
            if len(paths) > max_per_class:
                paths = paths[: max_per_class * 3]

            kept = 0
            class_vecs: list[np.ndarray] = []  # lista di vettori (1, D)

            for chunk in tqdm(list(batched(paths, batch_size)), desc=f"Indexing({lab})"):
                batch_imgs, batch_paths = [], []
                for p in chunk:
                    pth = Path(p)
                    if not pth.exists():
                        continue
                    try:
                        batch_imgs.append(Image.open(pth).convert("RGB"))
                        batch_paths.append(pth.as_posix())
                    except Exception:
                        # file corrotti/non leggibili
                        continue
                if not batch_imgs:
                    continue

                # Embedding uno-per-uno (più semplice per gestire TTA flip)
                for pil_img, pth in zip(batch_imgs, batch_paths):
                    try:
                        base_img = pil_img
                        # Opzionale: augment di luminanza solo se esplicitamente richiesto
                        if aug_index:
                            # Applica un gamma random moderato (es. 0.9–1.1)
                            g = float(np.clip(np.random.normal(1.0, 0.05), 0.85, 1.15))
                            base_img = gamma_pil(pil_img, g)

                        z_np = _embed_one(base_img)  # (1, D), già L2
                    except Exception:
                        continue

                    # Dedup intra-classe: se troppo simile, saltiamo
                    if class_vecs:
                        existing = np.vstack(class_vecs)  # (K, D)
                        # IP su vettori unitari = coseno
                        sims = existing @ z_np.squeeze(0)  # (K,)
                        if float(sims.max()) >= float(dedup_thr):
                            continue

                    # Accetta il campione
                    class_vecs.append(z_np)
                    all_embs.append(z_np)
                    rec = {"path": pth, "label": lab, "aug": ("gamma" if aug_index else "none")}
                    all_meta.append(rec)
                    fmeta.write(json.dumps(rec) + "\n")
                    kept += 1
                    if kept >= max_per_class:
                        break

                if kept >= max_per_class:
                    break

            print(f"[{lab}] kept={kept} (cap={max_per_class})  discarded={len(paths) - kept}")

    if not all_embs:
        raise RuntimeError("Nessun embedding calcolato. Controlla paths/manifest.")

    # Impila lungo l'asse batch: (N, D)
    embs: np.ndarray = np.concatenate(all_embs, axis=0)
    np.save(out_dir / "embeddings.npy", embs)

    # Statistiche rapide per report
    counts: Dict[str, int] = {}
    for rec in all_meta:
        counts[rec["label"]] = counts.get(rec["label"], 0) + 1
    (out_dir / "stats.json").write_text(
        json.dumps({"counts": counts, "total": int(embs.shape[0])}, indent=2),
        encoding="utf-8",
    )

    # (Opzionale) Costruzione indice FAISS flat/IP
    if faiss:
        build_faiss_index(out_dir, metric="ip", kind="flat")

    print(
        f"Salvati: {embs.shape[0]} embeddings di dimensione {embs.shape[1]} in {out_dir}"
    )
    print(f"Metadati: {meta_path}")
    print(f"Per-classe: {counts}")
    if faiss:
        print(f"FAISS index creato in {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="manifests/train.csv",
        type=str,
        help="Manifest path,label per l'indice",
    )
    ap.add_argument(
        "--out", default="index", type=str, help="Cartella output per embeddings e metadati"
    )
    ap.add_argument("--batch-size", default=64, type=int)

    # CLIP
    ap.add_argument("--model", default="ViT-B-32", type=str)
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k", type=str)
    ap.add_argument("--finetuned", default="", type=str, help="Path checkpoint fine-tuned (.pt).")

    # Switch backend + DINOv2
    ap.add_argument("--backend", choices=["clip", "dino"], default="clip")
    ap.add_argument("--dino-model", default="dinov2_vits14")

    # Augment di luminanza (sconsigliato per indice "neutro")
    ap.add_argument("--aug-index", action="store_true")

    # FAISS
    ap.add_argument("--faiss", action="store_true")

    # Politiche di qualità
    ap.add_argument("--max-per-class", type=int, default=120)
    ap.add_argument("--dedup-thr", type=float, default=0.998)
    ap.add_argument("--no-tta-flip", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    build_index(
        Path(args.manifest),
        Path(args.out),
        batch_size=args.batch_size,
        model_name=args.model,
        pretrained=args.pretrained,
        finetuned=(args.finetuned or None),
        aug_index=args.aug_index,
        backend=args.backend,
        dino_model=args.dino_model,
        faiss=args.faiss,
        max_per_class=args.max_per_class,
        dedup_thr=args.dedup_thr,
        tta_flip=not args.no_tta_flip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
