"""
segment_scene.py

Scopo
-----
Utility per la segmentazione automatica di scene complesse con SAM (Segment
Anything). Fornisce:

- `load_sam_automatic(...)` : costruisce un `SamAutomaticMaskGenerator` con
  iperparametri controllabili e workaround di caricamento pesi.
- `generate_masks(...)`      : genera maschere SAM, applica filtri leggeri e
  restituisce una lista ordinata di dict SAM (contenenti, tra le altre, le
  chiavi `'segmentation'`, `'bbox'`, `'area'`, `'stability_score'`, `'predicted_iou'`).
- `mask_to_bbox(seg)`        : ricava il bounding box tight (x1,y1,x2,y2) da una
  maschera booleana.

Note
----
- SAM richiede immagini RGB (quindi convertiamo da BGR a RGB).
- Gli iperparametri del generatore (densità campioni, threshold IoU/stability,
  piramide di crop) hanno impatto diretto su #maschere e qualità.
- `box_nms_thresh` viene impostato a 0.7 per ridurre duplicati spaziali.
- È supportato un fallback robusto per il caricamento pesi quando si usano
  versioni di PyTorch in cui `weights_only=True` può dare problemi.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# -----------------------------------------------------------------------------
# Costruttore del generatore automatico di maschere SAM
# -----------------------------------------------------------------------------

def load_sam_automatic(
    ckpt: str = "weights/sam_vit_b_01ec64.pth",
    model_type: str = "vit_b",
    points_per_side: int = 8,                # 24 => più denso, più lento
    pred_iou_thresh: float = 0.70,           # ↑ => maschere più conservative
    stability_score_thresh: float = 0.85,    # ↑ => filtra maschere instabili
    crop_n_layers: int = 1,                  # >0 => piramide di crop, più dettagli
    crop_n_points_downscale_factor: int = 2, # ↓ => più punti nei crop (più lento)
) -> SamAutomaticMaskGenerator:
    """Crea un `SamAutomaticMaskGenerator` per segmentare scene complesse.

    Parametri principali
    --------------------
    points_per_side : int
        Densità della griglia di punti sull'immagine. Più alto => più proposte.
    pred_iou_thresh : float
        Soglia minima sull'IoU predetto per accettare una proposta.
    stability_score_thresh : float
        Filtra maschere con score di stabilità basso (più fragile = più rumore).
    crop_n_layers : int
        Numero di layer nella piramide di crop (0 = disabilitato). Aumenta i
        dettagli su oggetti piccoli ma rallenta.
    crop_n_points_downscale_factor : int
        Fattore di downscale dei punti per i crop (più piccolo => più punti =>
        maggiore qualità/lentezza).

    Returns
    -------
    SamAutomaticMaskGenerator
        Generatore pronto a produrre una lista di dict SAM via `.generate(img)`.

    Note
    ----
    - Se il caricamento diretto del checkpoint fallisce con errori relativi a
      `weights_only`, si adotta un fallback che istanzia il modello e carica lo
      state_dict in modo esplicito.
    - Il modello viene spostato automaticamente su `cuda` se disponibile.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Tentativo "normale" (può fallire in combinazioni PyTorch/SAM recenti)
        sam = sam_model_registry[model_type](checkpoint=ckpt)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "weights_only" in msg:
            print("[SAM] Retry: torch.load(..., weights_only=False)")
            # Costruisci il modello senza caricare i pesi
            sam = sam_model_registry[model_type](checkpoint=None)
            # Ricarica il checkpoint in modo esplicito (file trusted)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            sam.load_state_dict(state, strict=True)
        else:
            raise

    sam.to(device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        box_nms_thresh=0.7,
    )


# -----------------------------------------------------------------------------
# Generazione e filtraggio leggero delle maschere
# -----------------------------------------------------------------------------

def generate_masks(
    image_bgr: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    min_area: int = 1200,
    max_masks: Optional[int] = None,
    sort_by: str = "area",          # "area" oppure "stability_score"
) -> List[Dict]:
    """Genera maschere SAM e applica filtri leggeri.

    Parameters
    ----------
    image_bgr : np.ndarray
        Immagine in formato BGR (OpenCV). Verrà convertita in RGB per SAM.
    mask_generator : SamAutomaticMaskGenerator
        Generatore creato da `load_sam_automatic`.
    min_area : int
        Scarta segmenti molto piccoli (rumore). Espresso in pixel.
    max_masks : Optional[int]
        Se impostato, limita il numero di maschere restituite dopo
        l'ordinamento.
    sort_by : {"area", "stability_score"}
        Criterio di ordinamento decrescente (default: area).

    Returns
    -------
    List[Dict]
        Lista di dict generati da SAM (chiavi tipiche: 'segmentation' (bool),
        'bbox', 'area', 'stability_score', 'predicted_iou').

    Note
    ----
    - SAM lavora su RGB: qui convertiamo da BGR (OpenCV) a RGB.
    - `min_area` è un filtro pratico per ridurre il rumore su dettagli minuscoli.
    """
    # SAM lavora in RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    raw = mask_generator.generate(image_rgb)

    # filtro area minima
    masks = [m for m in raw if int(m.get("area", 0)) >= int(min_area)]

    # ordinamento
    if sort_by == "stability_score":
        masks.sort(key=lambda d: float(d.get("stability_score", 0.0)), reverse=True)
    else:
        masks.sort(key=lambda d: int(d.get("area", 0)), reverse=True)

    if max_masks is not None:
        masks = masks[:max_masks]
    return masks


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def mask_to_bbox(seg: np.ndarray) -> Tuple[int, int, int, int]:
    """Restituisce il bounding box tight (x1, y1, x2, y2) da una maschera binaria.

    Se `seg` non contiene pixel attivi, ritorna (0, 0, 0, 0).
    """
    ys, xs = np.where(seg)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return x1, y1, x2, y2
