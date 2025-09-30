"""
dino_backend.py

Scopo
-----
Fornisce una piccola API per:
1) caricare un modello DINOv2 tramite `torch.hub` e la relativa pipeline di
   preprocess in stile ImageNet;
2) ottenere l'embedding L2-normalizzato di una PIL Image.

Note
----
- Gli embedding sono unit norm (L2=1): il prodotto interno corrisponde alla
  cosine similarity. Questo li rende compatibili con ricerche FAISS/IP e
  con metriche coseno.
- Le trasformazioni seguono lo standard ImageNet: `Resize→CenterCrop 224→Tensor→Normalize`.
- Il modello viene messo in `eval()` e usato con `@torch.no_grad()` per evitare
  di tracciare il grafo (più efficiente in inference).
"""

import torch
import torchvision.transforms as T
from PIL import Image

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(model_name: str = "dinov2_vits14", device=None):
    """Carica il modello DINOv2 e la pipeline di preprocess.

    Parametri
    ---------
    model_name : str
        Nome del checkpoint DINOv2 da `facebookresearch/dinov2` (es. "dinov2_vits14").
    device : str | None
        Dispositivo su cui posizionare il modello ("cuda"/"cpu"). Se `None`,
        viene scelto automaticamente in base alla disponibilità di CUDA.

    Ritorna
    -------
    (model, preprocess, device)
        - `model`     : nn.Module in `eval()`
        - `preprocess`: `torchvision.transforms.Compose` compatibile con PIL
        - `device`    : stringa del device effettivo
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Caricamento tramite torch.hub (repo ufficiale DINOv2)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()

    # Preprocess standard (ImageNet-like)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return model, preprocess, device


@torch.no_grad()
def embed_pil_dino(pil_img: Image.Image, model, preprocess, device):
    """Restituisce l'embedding DINOv2 L2-normalizzato di una PIL Image.

    Passi
    -----
    1) Converte l'immagine in RGB (per sicurezza)
    2) Applica il `preprocess` (Resize→CenterCrop 224→Normalize)
    3) Aggiunge la dimensione batch e sposta su `device`
    4) Forward del modello (shape `(1, D)`)
    5) Normalizza L2 lungo la dimensione dei canali (`dim=-1`)
    6) Trasferisce su CPU e ritorna

    Parametri
    ---------
    pil_img : PIL.Image.Image
        Immagine sorgente in formato PIL (qualsiasi size); verrà convertita in RGB.
    model : torch.nn.Module
        Modello DINOv2 caricato via `load_dino`.
    preprocess : callable
        Pipeline di trasformazioni (da `load_dino`).
    device : str
        Dispositivo ("cuda"/"cpu").

    Ritorna
    -------
    torch.Tensor
        Tensore di shape `(1, D)`, L2-normalizzato, su CPU.
    """
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    z = model(x)  # (1, D)
    z = torch.nn.functional.normalize(z, dim=-1)
    return z.cpu()  # torch.Tensor (1, D), unit-norm
