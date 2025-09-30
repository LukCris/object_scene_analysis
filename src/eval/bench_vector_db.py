"""
bench_vector_db.py

Scopo
-----
Misura velocità e fedeltà (parità top-1) tra:
- NAIVE: prodotto interno denso (cosine su vettori L2) su tutta la matrice indice
- FAISS: `IndexFlatIP` (inner product) su CPU

Assunzioni
---------
- Gli embedding in `embeddings.npy` sono L2-normalizzati.
  In tal caso, `inner product` ≡ cosine similarity.
- `faiss_db.build_faiss_index` ha salvato un indice IP/flat.

Output
------
- Tempo totale e per-query per NAIVE e FAISS.
- Top-1 parity: percentuale di query per cui il miglior risultato naïve
  coincide col miglior risultato FAISS (deve essere ~100% con FlatIP).
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.search.segment_matcher import load_index
from src.search.faiss_db import (
    build_faiss_index,
    faiss_exists,
    load_faiss_index,
    search_faiss,
)


def cosine_topk_naive(Q: np.ndarray, M: np.ndarray | torch.Tensor, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k coseno via matmul denso (naive).

    Parametri
    ---------
    Q : (B, D) numpy
        Batch di query **unit norm**.
    M : (N, D) numpy o torch.Tensor
        Matrice indice **unit norm**.
    k : int
        Numero di vicini da restituire.

    Ritorna
    -------
    (vals_sorted, idx_sorted)
        - `vals_sorted`: (B, k) cosine similarities ordinate decrescenti
        - `idx_sorted` : (B, k) indici dei top-k in `M`
    """
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    # prodotto interno = coseno su vettori normalizzati
    sims = Q @ M.T  # (B, N)

    # argpartition per prendere i top-k grezzi, poi ordina per ciascuna riga
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(Q.shape[0])[:, None]
    idx_sorted = idx[rows, np.argsort(-sims[rows, idx], axis=1)]
    vals_sorted = np.take_along_axis(sims, idx_sorted, axis=1)
    return vals_sorted, idx_sorted


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark naive vs FAISS (FlatIP)")
    ap.add_argument("--index", default="index_ft")
    ap.add_argument("--build-faiss", action="store_true", help="Forza la (ri)costruzione dell'indice FAISS")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    index_dir = Path(args.index)
    mat, meta = load_index(index_dir)  # mat: torch (N, D) L2-normalized
    N, D = mat.shape

    if args.build_faiss or not faiss_exists(index_dir):
        build_faiss_index(index_dir, metric="ip", kind="flat")
    faiss_idx = load_faiss_index(index_dir)

    # Query: campiona senza rimpiazzo embedding dall'indice
    B = min(args.queries, N)
    pick = np.random.choice(N, size=B, replace=False)
    Q = mat[pick].numpy()  # (B, D)

    # --- NAIVE ---
    t0 = time.time()
    vals_naive, idx_naive = cosine_topk_naive(Q, mat.numpy(), k=args.k)
    t1 = time.time()

    # --- FAISS ---
    t2 = time.time()
    vals_faiss, idx_faiss = search_faiss(faiss_idx, Q, k=args.k)
    t3 = time.time()

    # Parità top-1
    top1_match = float((idx_naive[:, 0] == idx_faiss[:, 0]).mean())

    print(f"[INDEX] N={N} D={D}  |  B={B}  k={args.k}")
    print(f"[NAIVE]  total={t1 - t0:.4f}s   per_query={(t1 - t0) / B * 1000:.2f} ms")
    print(f"[FAISS]  total={t3 - t2:.4f}s   per_query={(t3 - t2) / B * 1000:.2f} ms")
    print(f"Top-1 parity (naive vs faiss): {top1_match * 100:.2f}%")


if __name__ == "__main__":
    main()
