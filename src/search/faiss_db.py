"""
faiss_db.py

Scopo
-----
Utility per creare, salvare, caricare e interrogare un indice FAISS a partire
da `embeddings.npy` (matrice (N, D)). Il caso d'uso principale è la ricerca per
similarità coseno quando i vettori sono L2-normalizzati, usando un indice
`flat` con inner product (IP).

File generati in `<index_dir>`
------------------------------
- `faiss.index`      : indice FAISS serializzato
- `faiss_meta.json`  : metadati (metric, kind, dimensione, #vettori)
"""

import json, faiss

from pathlib import Path
from typing import Tuple

import numpy as np

FAISS_FILE = "faiss.index"
FAISS_META = "faiss_meta.json"


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

def build_faiss_index(index_dir: Path, metric: str = "ip", kind: str = "flat") -> None:
    """Costruisce un indice FAISS da `embeddings.npy` (N, D).

    Parameters
    ----------
    index_dir : Path
        Cartella che contiene `embeddings.npy` (float32/float64; verrà castato).
    metric : {"ip", "l2"}
        Metrica per la ricerca: inner product (IP) o distanza L2.
    kind : {"flat"}
        Tipo di indice (solo `flat` implementato).

    Side effects
    ------------
    Salva `faiss.index` e `faiss_meta.json` dentro `index_dir`.
    """
    index_dir = Path(index_dir)
    embs = np.load(index_dir / "embeddings.npy")  # (N, D)
    if embs.ndim != 2:
        raise ValueError("embeddings.npy deve essere 2D (N, D)")
    N, D = embs.shape

    # Selezione metrica
    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric deve essere 'ip' o 'l2'")

    # Selezione indice
    if kind == "flat":
        idx = faiss.IndexFlatIP(D) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(D)
    else:
        raise NotImplementedError("Solo 'flat' implementato per semplicità.")

    # Aggiungi vettori (float32).
    idx.add(embs.astype(np.float32, copy=False))

    # Persistenza
    faiss.write_index(idx, (index_dir / FAISS_FILE).as_posix())
    (index_dir / FAISS_META).write_text(
        json.dumps({"metric": metric, "kind": kind, "dim": int(D), "size": int(N)}, indent=2),
        encoding="utf-8",
    )
    print(f"[FAISS] scritto indice {kind}/{metric} con N={N}, D={D} -> {index_dir/FAISS_FILE}")


# -----------------------------------------------------------------------------
# Exists / Load
# -----------------------------------------------------------------------------

def faiss_exists(index_dir: Path) -> bool:
    """Ritorna True se `faiss.index` è presente in `index_dir`."""
    return (Path(index_dir) / FAISS_FILE).exists()


def load_faiss_index(index_dir: Path):
    """Carica un indice FAISS da `faiss.index` in `index_dir`.

    Raises
    ------
    FileNotFoundError
        Se il file indice non è presente.
    """
    p = (Path(index_dir) / FAISS_FILE).as_posix()
    if not Path(p).exists():
        raise FileNotFoundError(p)
    idx = faiss.read_index(p)
    return idx


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

def search_faiss(idx, q: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Esegue una ricerca top-k su un indice FAISS.

    Parameters
    ----------
    idx : faiss.Index
        Indice FAISS già caricato.
    q : np.ndarray, shape (D,) o (B, D), dtype float32
        Query vector(s). Verranno castati a float32 e reshaped se necessario.
    k : int
        Numero di vicini da restituire.

    Returns
    -------
    (scores, ids)
        `scores`: (B, k) — IP o -L2 (a seconda della metrica)
        `ids`   : (B, k) — indici nel dataset indicizzato

    Note
    ----
    - Con `metric='ip'` e vettori unitari, `scores` è la cosine similarity.
    """
    if q.ndim == 1:
        q = q[None, :]
    q = q.astype(np.float32, copy=False)
    scores, ids = idx.search(q, k)
    return scores, ids
