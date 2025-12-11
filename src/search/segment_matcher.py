"""
segment_matcher.py

Scopo
-----
Moduli per:
- Caricare un backend di embedding (CLIP oppure DINOv2).
- Ottenere un embedding da una `PIL.Image` (L2-normalizzato).
- Caricare un indice (embeddings + meta) salvato dalla fase di indicizzazione.
- Effettuare matching tramite similarità coseno naive oppure via
  FAISS se disponibile (flat/IP).

Output tipico per un segmento:
- `decide_label_baseline(...)` -> (label_top1, score, {dbg})
- `decide_label_faiss(...)`    -> (label_top1, score, {dbg})
"""


import json, torch, open_clip

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from pathlib import Path

from src.search.faiss_db import faiss_exists, load_faiss_index, search_faiss


# -----------------------------------------------------------------------------
# Costanti preprocess (DINO)
# -----------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _to_numpy(x):
    """Converte `torch.Tensor` -> `np.ndarray` (CPU, detach), altrimenti ritorna x.

    Utile quando alcune API (es. FAISS) richiedono `float32` NumPy.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# -----------------------------------------------------------------------------
# Backend loading & embedding
# -----------------------------------------------------------------------------

def load_backend(
    backend: str = "clip",
    clip_model: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    dino_model: str = "dinov2_vits14",
    device: str | None = None,
    finetuned: str | None = None,
):
    """Carica il backend di embedding e il relativo preprocess.

    Parameters
    ----------
    backend : {"clip", "dino"}
        Selezione del modello di embedding.
    clip_model : str
        Architettura CLIP per `open_clip` (es. "ViT-B-32").
    pretrained : str
        Tag del checkpoint pre-addestrato per `open_clip`.
    dino_model : str
        Nome del modello DINOv2 per `torch.hub` (es. "dinov2_vits14").
    device : Optional[str]
        Dispositivo su cui caricare il modello (default: auto).
    finetuned : Optional[str]
        Path a un checkpoint di CLIP fine-tuned; caricato in `strict=False`.

    Returns
    -------
    tuple
        (kind, model, preprocess, device) dove `kind` in {"clip", "dino"}.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "clip":
        # Carico modello CLIP con stessi parametri del training
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        preprocess = preprocess_val
        model = model.to(device).eval()

        if finetuned:
            ckpt = torch.load(finetuned, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[finetuned] loaded: {finetuned}")
            print(f"   missing={len(missing)}  unexpected={len(unexpected)}")

        return "clip", model, preprocess, device

    else:
        # Backend DINOv2
        model = torch.hub.load("facebookresearch/dinov2", dino_model)
        model = model.to(device).eval()
        preprocess = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        return "dino", model, preprocess, device


def embed_pil_for_clip(
    pil_img: Image.Image, model, preprocess, device: str
) -> torch.Tensor:
    """Calcola embedding CLIP L2-normalizzato da una PIL Image.

    Returns
    -------
    torch.Tensor
        Tensore (1, D) su CPU.
    """
    with torch.no_grad():
        x = preprocess(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)
        z = model.encode_image(x)  # (1, D)
        z = F.normalize(z, dim=-1)
    return z.cpu()


@torch.no_grad()
def embed_pil_dino(pil_img: Image.Image, model, preprocess, device: str) -> torch.Tensor:
    """Calcola embedding DINOv2 L2-normalizzato da una PIL Image."""
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    z = model(x)  # (1, D)
    z = F.normalize(z, dim=-1)
    return z.cpu()


def embed_pil_generic(pil_img: Image.Image, backend_tuple) -> torch.Tensor:
    """Wrapper che instrada verso la funzione di embedding corretta (CLIP/DINO)."""
    kind, model, preprocess, device = backend_tuple
    if kind == "clip":
        return embed_pil_for_clip(pil_img, model, preprocess, device)
    else:
        return embed_pil_dino(pil_img, model, preprocess, device)


# -----------------------------------------------------------------------------
# FAISS (cache + search)
# -----------------------------------------------------------------------------

_faiss_cache = {"dir": None, "idx": None}


def get_faiss(index_dir: Path):
    """Ritorna un handle FAISS per `index_dir` con caching in-process.

    Se l'indice non esiste, ritorna `None`.
    """
    global _faiss_cache
    if _faiss_cache["dir"] == str(index_dir) and _faiss_cache["idx"] is not None:
        return _faiss_cache["idx"]
    if not faiss_exists(index_dir):
        return None
    idx = load_faiss_index(index_dir)
    _faiss_cache = {"dir": str(index_dir), "idx": idx}
    return idx


# -----------------------------------------------------------------------------
# Index I/O
# -----------------------------------------------------------------------------

def load_index(index_dir: Path) -> tuple[torch.Tensor, list[dict]]:
    """Carica `embeddings.npy` (N, D) e `meta.jsonl` (list di dict).

    Returns
    -------
    (index_mat, id2meta)
        `index_mat` = `torch.FloatTensor` (N, D) su CPU, già L2-normalizzato.
        `id2meta`    = lista di dict con almeno `label`.

    Raises
    ------
    ValueError
        Se `embs.shape[0]` e `len(meta)` non coincidono.
    """
    embs = np.load(index_dir / "embeddings.npy")  # (N, D)
    lines = (index_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()
    meta = [json.loads(l) for l in lines]
    if embs.shape[0] != len(meta):
        raise ValueError("embeddings.npy e meta.jsonl hanno dimensioni diverse.")
    index_mat = torch.from_numpy(embs).float()  # già L2-normalized
    return index_mat, meta


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

def topk_cosine(z: torch.Tensor, index_mat: torch.Tensor, k: int = 5) -> tuple[list[float], list[int]]:
    """Top-k per similarità coseno tra `z` (1, D) e `index_mat` (N, D).

    Returns
    -------
    (vals, idx)
        `vals` = lista dei top-k score coseno
        `idx`  = lista degli indici corrispondenti in `index_mat`
    """
    sims = (z @ index_mat.T).squeeze(0)  # (N,)
    vals, idx = torch.topk(sims, k=min(k, sims.numel()))
    return vals.detach().cpu().numpy().tolist(), idx.detach().cpu().numpy().tolist()


def decide_label_baseline(
    z, index_mat: torch.Tensor, id2meta: list[dict]
) -> tuple[str, float, dict]:
    """Predizione baseline: etichetta del nearest neighbor (top-1 coseno).

    Parameters
    ----------
    z : torch.Tensor | np.ndarray
        Embedding (1, D) o (D,). Se (D,), viene aggiunta una dimensione batch.
    index_mat : torch.Tensor (N, D)
        Matrice degli embedding indice, già L2-normalizzati.
    id2meta : list[dict]
        Metadati (almeno `label`) allineati con `index_mat`.

    Returns
    -------
    (label, score, dbg)
        `label` = etichetta del nearest neighbor
        `score` = similarità coseno
        `dbg`   = info di debug (mode, indice selezionato)
    """
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()
    if z.ndim == 1:
        z = z.unsqueeze(0)

    sims = (z @ index_mat.T).squeeze(0)  # (N,)
    j = int(torch.argmax(sims))
    label = id2meta[j]["label"]
    score = float(sims[j])
    return label, score, {"mode": "baseline", "j": j}


def decide_label_faiss(z, index_dir: Path, meta: list[dict]) -> tuple[str, float, dict]:
    """Predizione via FAISS (flat/IP) su vettori L2-normalizzati.

    Parameters
    ----------
    z : torch.Tensor | np.ndarray
        Embedding (1, D) o (D,). Viene convertito a `np.float32` per FAISS.
    index_dir : Path
        Cartella dell'indice FAISS.
    meta : list[dict]
        Metadati allineati con l'indice.

    Returns
    -------
    (label, score, dbg)
        `label` = etichetta del nearest neighbor
        `score` = similarità IP (equiv. coseno per vettori unitari)
        `dbg`   = info di debug (mode, indice selezionato)
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    idx = get_faiss(index_dir)
    assert idx is not None, "Indice FAISS non trovato"
    scores, ids = search_faiss(idx, z.astype(np.float32), k=1)
    j = int(ids[0, 0])
    sc = float(scores[0, 0])
    label = meta[j]["label"]
    return label, sc, {"mode": "faiss", "j": j}
