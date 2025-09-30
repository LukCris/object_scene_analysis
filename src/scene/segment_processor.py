"""
segment_processor.py

Scopo
-----
Utility per post-processare le maschere prodotte da SAM su scene complesse e
preparare ritagli compatibili con CLIP/DINO:

- `apply_mask_rgba(...)`   : applica la maschera e produce un'immagine BGRA
  (alpha=255 dentro la maschera, 0 altrove).
- `crop_mask_tight(...)`   : ritaglia sul tight bbox (con pad) mantenendo
  l'alpha.
- `rgba_to_rgb_for_clip(...)` : compone un RGB (PIL) adatto al preprocess CLIP;
  se presente alpha, effettua compositing su un background uniforme.

Note
----
- OpenCV usa BGR; PIL usa RGB → la conversione è gestita quando necessario.
- Le funzioni assumono dtype `uint8` e shape standard (H,W,3|4) per immagini,
  (H,W[,(1)]) per maschere. Gli `assert` aiutano a fallire presto con messaggio
  chiaro.
"""

import cv2
import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Applica maschera -> BGRA
# -----------------------------------------------------------------------------

def apply_mask_rgba(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Applica una maschera binaria a un'immagine BGR e restituisce un BGRA.

    Canali in uscita:
    - B,G,R = immagine originale
    - A     = 255 dove `seg=True`, 0 altrove

    Parameters
    ----------
    image_bgr : np.ndarray (H, W, 3), uint8
        Immagine BGR (OpenCV).
    seg : np.ndarray (H, W) o (H, W, 1), bool oppure {0,1}
        Maschera oggetto (True/1 = oggetto).

    Returns
    -------
    np.ndarray (H, W, 4), uint8
        Immagine BGRA con alpha "ritagliato".
    """
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3, "image_bgr deve essere (H,W,3)"
    seg_bool = seg.astype(bool)
    h, w = image_bgr.shape[:2]
    assert seg_bool.shape[:2] == (h, w), "Maschera e immagine devono avere stesse dimensioni"

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[seg_bool] = 255
    bgra[:, :, 3] = alpha
    return bgra


# -----------------------------------------------------------------------------
# Ritaglio tight bbox con pad (mantiene alpha)
# -----------------------------------------------------------------------------

def crop_mask_tight(image_bgra: np.ndarray, seg: np.ndarray, pad: int = 8) -> np.ndarray:
    """Ritaglia sul bounding box tight della maschera, con padding e clamp ai bordi.

    Il canale alpha (se presente) viene mantenuto. Se la maschera è vuota,
    restituisce l'input (fallback sicuro).

    Note: il bbox è inclusivo; per slicing numpy usiamo y2+1/x2+1.

    Parameters
    ----------
    image_bgra : np.ndarray (H, W, 3|4), uint8
        Immagine BGR o BGRA.
    seg : np.ndarray (H, W) o (H, W, 1)
        Maschera booleana o binaria.
    pad : int
        Padding (in pixel) applicato su tutti i lati.

    Returns
    -------
    np.ndarray (h', w', 3|4), uint8
        Ritaglio dell'immagine originale.
    """
    assert image_bgra.ndim == 3 and image_bgra.shape[2] in (3, 4), "Atteso BGR o BGRA"
    seg_bool = seg.astype(bool)
    ys, xs = np.where(seg_bool)
    if xs.size == 0 or ys.size == 0:
        # Nessun pixel positivo: ritorna input (fallback sicuro)
        return image_bgra

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # padding ai bordi
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(image_bgra.shape[1] - 1, x2 + pad)
    y2 = min(image_bgra.shape[0] - 1, y2 + pad)
    return image_bgra[y1 : y2 + 1, x1 : x2 + 1, :]


# -----------------------------------------------------------------------------
# Compositing RGBA -> RGB (PIL) per CLIP/DINO
# -----------------------------------------------------------------------------

def rgba_to_rgb_for_clip(image_bgra: np.ndarray, bg_value: int = 255) -> Image.Image:
    """Converte BGR/BGRA (OpenCV) in RGB (PIL) pronto per CLIP/DINO.

    Se presente un canale alpha, compone su un background uniforme `bg_value`.

    Parameters
    ----------
    image_bgra : np.ndarray (H, W, 3|4), uint8
        Immagine BGR o BGRA.
    bg_value : int
        Valore [0..255] per le zone trasparenti.

    Returns
    -------
    PIL.Image.Image (RGB)
    """
    assert image_bgra.ndim == 3 and image_bgra.shape[2] in (3, 4), "Atteso BGR o BGRA uint8"

    if image_bgra.shape[2] == 3:
        # Nessun alpha: converti BGR -> RGB
        rgb = cv2.cvtColor(image_bgra, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # Con alpha: compositing su colore uniforme
    bgr = image_bgra[:, :, :3].astype(np.float32)
    alpha = (image_bgra[:, :, 3:4].astype(np.float32)) / 255.0
    bg = np.full_like(bgr, fill_value=float(bg_value), dtype=np.float32)
    comp = (bgr * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
