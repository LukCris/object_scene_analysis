"""
train_clip_finetune.py

Scopo
-----
Fine-tuning (leggero) di CLIP su un dataset custom con manifest `path,label`.
Alleniamo la torre visiva a discriminare le classi, usando prototipi testuali
(calcolati da prompt/alias) come classifier (cosine → logit con `logit_scale`).

Pipeline
--------
1) Carica CLIP (open_clip) + preprocess train/val + tokenizer
2) Prepara i Dataset da manifest CSV (train/val)
3) Costruisce i prototipi testuali (media su template × alias per classe)
4) Train loop: `CrossEntropy(img @ text.T * exp(logit_scale))` con opz.:
   - class weights (bilanciamento), label smoothing
   - gradient accumulation, AMP, grad clip, early stopping
5) Valuta su validation, salva checkpoint/best e metriche, confusion matrix

Note importanti
---------------
- Gli embedding immagine/testo sono L2-normalizzati -> prodotto interno ≡ coseno.
- Se `--freeze-text`, i prototipi testuali vengono precomputati e mantenuti fissi.
"""


import argparse, json, os, time, open_clip, torch

from dataclasses import dataclass
from pathlib import Path
from typing import List
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ManifestImageDataset(torch_data.Dataset):
    """Dataset da manifest CSV `path,label`.

    - Usa il `preprocess` restituito da `open_clip.create_model_and_transforms`.
    - Assume che i path siano validi e le label appartengano a `classes`.
    """

    def __init__(self, manifest_csv: str | Path, classes: List[str], preprocess):
        self.manifest_csv = Path(manifest_csv)
        df = pd.read_csv(self.manifest_csv)
        # normalizza header
        cols = [c.strip().lower() for c in df.columns]
        assert (
            cols == ["path", "label"]
        ), f"Manifest {manifest_csv} deve avere header 'path,label', got={df.columns.tolist()}"
        self.items = [
            (str(p), str(l)) for p, l in zip(df["path"].astype(str), df["label"].astype(str))
        ]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        img = Image.open(p).convert("RGB")
        x = self.preprocess(img)  # (3,H,W), float tensor
        try:
            y = self.class_to_idx[lab]
        except KeyError as e:
            raise KeyError(
                f"Label '{lab}' non presente in classes. Controlla {self.manifest_csv} e classes.json"
            ) from e
        return x, y


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_seed(s: int) -> None:
    import random
    import numpy as _np

    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    _np.random.seed(s)


@dataclass
class Batch:
    images: torch.Tensor  # (B,3,H,W)
    targets: torch.Tensor  # (B,)


def build_text_features(
    classes: List[str],
    model,
    tokenizer,
    device,
    prompt_templates: List[str] | None = None,
    aliases_map: dict | None = None,
):
    """Crea i prototipi testuali facendo la media su (template × alias) per classe.

    Per ogni classe:
      1) genera una lista di frasi (alias × prompt template)
      2) tokenizza e codifica con `model.encode_text`
      3) L2-normalizza ogni embedding testo e ne fa la media
      4) rinormalizza il prototipo risultante

    Restituisce: `torch.Tensor (C, D)` su `device`.
    """
    if prompt_templates is None:
        prompt_templates = [
            "{}",
            "{} from Naruto",
            "an anime portrait of {}",
            "official character art of {}",
            "a cosplay photo of {}",
        ]

    # alias di default per le 4 classi del progetto
    default_aliases = {
        "Gara": ["Gara", "Gaara"],
        "Naruto": ["Naruto", "Uzumaki Naruto"],
        "Sakura": ["Sakura", "Haruno Sakura"],
        "Tsunade": ["Tsunade", "Lady Tsunade"],
    }
    if aliases_map:
        for k, v in aliases_map.items():
            default_aliases[k] = v

    feats = []
    with torch.no_grad():
        for cls in classes:
            names = default_aliases.get(cls, [cls])
            texts = [tmpl.format(name) for name in names for tmpl in prompt_templates]
            toks = tokenizer(texts).to(device)
            txt = model.encode_text(toks)  # (T,D)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            proto = txt.mean(dim=0)
            proto = proto / proto.norm()
            feats.append(proto)
    feats = torch.stack(feats, dim=0).to(device)  # (C,D)
    return feats


def compute_class_weights(train_manifest: str | Path, classes: List[str]) -> torch.Tensor:
    """Pesi di classe (inverso della frequenza, normalizzato) per CE bilanciata."""
    df = pd.read_csv(train_manifest)
    counts = {c: 0 for c in classes}
    for l in df["label"].astype(str).tolist():
        if l in counts:
            counts[l] += 1
    arr = np.array([counts[c] for c in classes], dtype=np.float32)
    arr = np.maximum(arr, 1.0)  # evita div/0
    inv_freq = 1.0 / arr
    w = inv_freq / inv_freq.mean()
    return torch.tensor(w, dtype=torch.float32)


def save_checkpoint(out_dir: Path, epoch: int, model, optimizer, scaler, best: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    p = out_dir / ("best.pt" if best else f"epoch_{epoch:03d}.pt")
    torch.save(obj, p)
    return p


def _count_trainable(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

def train_epoch(
    model,
    data: torch_data.DataLoader,
    device,
    text_feats,
    class_weights: torch.Tensor | None,
    optimizer,
    scaler,
    logit_scale,
    accum_steps: int = 1,
    label_smoothing: float = 0.0,
    clip_grad: float | None = 1.0,
):
    """Un'epoca di addestramento con CE bilanciata e gradient accumulation."""
    model.train()
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing,
    )
    total_loss = 0.0
    correct = 0
    total = 0

    params = [p for p in model.parameters() if p.requires_grad]

    for step, (images, targets) in enumerate(data, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            img = model.encode_image(images)  # (B,D)
            img = img / img.norm(dim=-1, keepdim=True)
            logits = (logit_scale.exp() * img @ text_feats.t())  # (B,C)
            loss = loss_fn(logits, targets) / accum_steps

        (scaler.scale(loss) if scaler is not None else loss).backward()

        if step % accum_steps == 0:
            if clip_grad is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item() * accum_steps)
        pred = logits.argmax(dim=-1)
        correct += int((pred == targets).sum().item())
        total += int(targets.numel())

    avg_loss = total_loss / max(1, len(data))
    acc = (correct / total) if total else 0.0
    return avg_loss, acc


# -----------------------------------------------------------------------------
# Eval
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data: torch_data.DataLoader, device, text_feats, logit_scale, return_loss: bool = True):
    """Valutazione su validation (opzionalmente calcola anche la loss)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss() if return_loss else None
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in data:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            img = model.encode_image(images)
            img = img / img.norm(dim=-1, keepdim=True)
            logits = (logit_scale.exp() * img @ text_feats.t())
            if return_loss:
                total_loss += float(loss_fn(logits, targets).item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == targets).sum().item())
        total += int(targets.numel())

    acc = (correct / total) if total else 0.0
    return (total_loss / max(1, len(data)), acc) if return_loss else acc


@torch.no_grad()
def eval_confusion_and_per_class(
    model, data: torch_data.DataLoader, device, text_feats, logit_scale, num_classes: int
):
    """Costruisce confusion matrix e accuracy per-classe (no sklearn)."""
    model.eval()
    y_true, y_pred = [], []
    for images, targets in data:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            img = model.encode_image(images)
            img = img / img.norm(dim=-1, keepdim=True)
            logits = (logit_scale.exp() * img @ text_feats.t())
        pred = logits.argmax(dim=-1)
        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_acc = {}
    for c in range(num_classes):
        denom = cm[c].sum()
        per_class_acc[c] = (cm[c, c] / denom) if denom > 0 else 0.0
    return cm, per_class_acc, y_true, y_pred


# -----------------------------------------------------------------------------
# Main (CLI)
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tuning contrastivo (image -> text CE) di CLIP su manifest."
    )
    ap.add_argument("--train-manifest", default="manifests/train.csv")
    ap.add_argument("--val-manifest", default="manifests/valid.csv")
    ap.add_argument("--classes", default="manifests/classes.json", help="Ordine canonico etichette.")
    ap.add_argument("--out", default="runs/clip_ft_v1")

    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation.")

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--freeze-text", action="store_true", help="Congela il ramo testuale; di default è disattivato.")
    ap.add_argument("--freeze-vision", action="store_true", default=False)

    ap.add_argument("--prompt", default="an anime portrait of {} from Naruto")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience on val_loss.")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--max-logit-scale", type=float, default=100.0, help="Clamp for exp(logit_scale).")

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modello + preprocess / tokenizer
    # open_clip ritorna (model, preprocess_train, preprocess_val)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # (opz) freeze: default allena torre visiva; se `freeze_vision` -> solo logit_scale
    if args.freeze_text:
        # Congela tutto tranne la torre visiva e logit_scale
        for name, p in model.named_parameters():
            if name.startswith("visual.") or name == "logit_scale":
                p.requires_grad = True
            else:
                p.requires_grad = False
    if args.freeze_vision:
        for p in model.visual.parameters():
            p.requires_grad = False  # in questo caso si allena solo logit_scale

    print(f"[freeze] trainable params = {_count_trainable(model)}")

    # Dati
    classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    train_ds = ManifestImageDataset(args.train_manifest, classes, preprocess_train)
    val_ds = ManifestImageDataset(args.val_manifest, classes, preprocess_val)

    # class weights per sbilanciamento
    class_weights = compute_class_weights(args.train_manifest, classes)

    num_workers = 2 if os.name == "nt" else 4  # su Windows meglio più bassi
    pin = torch.cuda.is_available()
    train_loader = torch_data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = torch_data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )

    # Ottimizzazione
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Salva anche le classi
    (out_dir / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Precompute text feature se il ramo testo è congelato, riusandoli ogni volta
    text_feats_fixed = None
    if args.freeze_text:
        text_feats_fixed = build_text_features(classes, model, tokenizer, device)

    max_lscale = float(args.max_logit_scale)

    best_acc = 0.0
    best_val_loss = float("inf")
    bad = 0
    patience = args.patience

    metrics = []  # per salvare le curve

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # (R) text feats per epoca (a meno di freeze)
        text_feats = (
            text_feats_fixed
            if text_feats_fixed is not None
            else build_text_features(classes, model, tokenizer, device)
        )
        logit_scale = model.logit_scale

        # train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            device,
            text_feats,
            class_weights,
            optimizer,
            scaler,
            logit_scale,
            accum_steps=args.accum_steps,
            label_smoothing=args.label_smoothing,
            clip_grad=1.0,
        )

        # clamp logit_scale con valore da CLI (stabilità numerica)
        with torch.no_grad():
            model.logit_scale.clamp_(max=np.log(max_lscale))

        # val
        val_loss, val_acc = evaluate(
            model, val_loader, device, text_feats, logit_scale, return_loss=True
        )

        dt = time.time() - t0
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)"
        )

        metrics.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "time_sec": float(dt),
            }
        )

        # checkpoint sempre
        save_checkpoint(out_dir, epoch, model, optimizer, scaler, best=False)

        # Best selection
        improved = (val_acc > best_acc) or (val_acc == best_acc and val_loss < best_val_loss)
        if improved:
            best_acc, best_val_loss, bad = val_acc, val_loss, 0
            p = save_checkpoint(out_dir, epoch, model, optimizer, scaler, best=True)
            print(f"  ↳ new BEST (val_acc={best_acc:.4f}, val_loss={best_val_loss:.4f}) → {p}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best_acc:.4f}.")
                break

    # salva metriche/curve
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix e per-class accuracy su validation
    num_classes = len(classes)
    text_feats_eval = (
        text_feats_fixed if text_feats_fixed is not None else build_text_features(classes, model, tokenizer, device)
    )
    cm, per_class_acc, y_true, y_pred = eval_confusion_and_per_class(
        model, val_loader, device, text_feats_eval, model.logit_scale, num_classes
    )

    # stampa leggibile
    print("\nValidation per-class accuracy:")
    for i, cls in enumerate(classes):
        print(f"  {cls:10s}: {per_class_acc[i]:.4f}")

    # salva su disco
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    pd.DataFrame({"class": classes, "per_class_acc": [per_class_acc[i] for i in range(num_classes)]}).to_csv(
        out_dir / "per_class_accuracy.csv", index=False
    )

    # plot (best-effort)
    try:
        epochs_x = [m["epoch"] for m in metrics]
        tr_loss = [m["train_loss"] for m in metrics]
        va_loss = [m["val_loss"] for m in metrics]
        tr_acc = [m["train_acc"] for m in metrics]
        va_acc = [m["val_acc"] for m in metrics]

        plt.figure(); plt.plot(epochs_x, tr_loss, label="train"); plt.plot(epochs_x, va_loss, label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curves"); plt.savefig(out_dir / "curves_loss.png", dpi=150); plt.close()
        plt.figure(); plt.plot(epochs_x, tr_acc, label="train"); plt.plot(epochs_x, va_acc, label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy curves"); plt.savefig(out_dir / "curves_acc.png", dpi=150); plt.close()

        plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (val)"); plt.colorbar();
        tick = np.arange(num_classes); plt.xticks(tick, classes, rotation=45, ha="right"); plt.yticks(tick, classes)
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        plt.tight_layout(); plt.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close()
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")

    print(f"\nDone. Best val_acc={best_acc:.4f}. Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
