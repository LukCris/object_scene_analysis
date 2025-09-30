from __future__ import annotations

"""
batch_analyze.py

Scopo
-----
Esegue la scene analysis in batch caricando una sola volta:
- backend di embedding (CLIP o DINOv2),
- indice (embeddings + meta),
- opzionalmente FAISS,
- e SAM (generatore automatico di maschere).

Questo elimina l'overhead di `analyze_scene.py` che ricarica tutto a ogni immagine.
L'API espone una classe `Analyzer` con `analyze_one(scene_path, out_dir, ...)` e
una CLI che processa un'intera cartella.

Dipendenze interne:
- src.sam_infer.segment_scene: load_sam_automatic, generate_masks, mask_to_bbox
- src.scene.segment_processor: apply_mask_rgba, crop_mask_tight, rgba_to_rgb_for_clip
- src.search.segment_matcher: load_backend, load_index, topk_cosine, decide_label_baseline
- src.search.faiss_db: faiss_exists, load_faiss_index, search_faiss
"""

import argparse, json, cv2

import numpy as np

from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from src.sam_infer.segment_scene import (
    load_sam_automatic,
    generate_masks,
    mask_to_bbox,
)
from src.scene.segment_processor import (
    apply_mask_rgba,
    crop_mask_tight,
    rgba_to_rgb_for_clip,
)
from src.search.segment_matcher import (
    load_backend,
    load_index,
    topk_cosine,
    decide_label_baseline,
)
from src.search.faiss_db import (
    faiss_exists,
    load_faiss_index,
    search_faiss,
)


# -----------------------------------------------------------------------------
# Utilities (overlay, NMS, selezione)
# -----------------------------------------------------------------------------

CLASS_COLORS = {
    "Naruto": (66, 133, 244),  # blu
    "Gara": (234, 67, 53),  # rosso
    "Sakura": (219, 39, 119),  # magenta
    "Tsunade": (52, 168, 83),  # verde
}
DEFAULT_COLOR = (255, 140, 0)  # arancione fallback


def draw_overlay(image_rgb: np.ndarray, boxes_labels_scores, out_path: Path) -> None:
    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2, lbl, sc) in boxes_labels_scores:
        color = CLASS_COLORS.get(lbl, DEFAULT_COLOR)
        drw.rectangle([(x1, y1), (x2, y2)], fill=color + (64,))
        drw.rectangle([(x1, y1), (x2, y2)], outline=color + (255,), width=2)

        text = f"{lbl} ({sc:.2f})"
        try:
            bbox = drw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = drw.textsize(text, font=font)

        pad = 2
        tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
        tx2, ty2 = x1 + tw + 2 * pad, y1
        drw.rectangle([(tx1, ty1), (tx2, ty2)], fill=(0, 0, 0, 200), outline=color + (255,), width=1)
        drw.text((tx1 + pad, ty1 + pad), text, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path.as_posix())


def _iou(a, b) -> float:
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


def nms_preds(boxes_labels_scores, iou_thr: float = 0.30):
    keep, used = [], [False] * len(boxes_labels_scores)
    order = sorted(range(len(boxes_labels_scores)), key=lambda i: boxes_labels_scores[i][5], reverse=True)
    for i in order:
        if used[i]:
            continue
        used[i] = True
        keep.append(boxes_labels_scores[i])
        for j in order:
            if used[j]:
                continue
            if _iou(boxes_labels_scores[i], boxes_labels_scores[j]) >= iou_thr:
                used[j] = True
    return keep


def select_topK_distinct(
    bls,
    K: int = 5,
    score_backoff=(0.80, 0.70, 0.60),
    iou_thr: float = 0.45,
    prefer_diff_label: bool = False,
    delta: float = 0.05,
):
    bls = sorted(bls, key=lambda t: t[5], reverse=True)
    for thr in score_backoff:
        cands = [b for b in bls if b[5] >= thr]
        picked = []
        for cand in cands:
            if any(_iou(cand, p) >= iou_thr for p in picked):
                continue
            if prefer_diff_label and len(picked) == 1 and cand[4] == picked[0][4]:
                alt = next(
                    (b for b in cands if b[4] != picked[0][4] and all(_iou(b, p) < iou_thr for p in picked)),
                    None,
                )
                if alt and (picked[0][5] - alt[5] <= delta or alt[5] >= thr):
                    cand = alt
            picked.append(cand)
            if len(picked) == K:
                return picked
    picked = []
    for cand in bls:
        if all(_iou(cand, p) < iou_thr for p in picked):
            picked.append(cand)
            if len(picked) == K:
                break
    return picked


def expand_bbox(x1, y1, x2, y2, H, W, ratio: float = 0.08):
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
    from src.search.segment_matcher import embed_pil_generic  # import locale per evitare dipendenze circolari
    z1 = embed_pil_generic(pil_rgb, backend_tuple)
    z2 = embed_pil_generic(pil_rgb.transpose(Image.FLIP_LEFT_RIGHT), backend_tuple)
    return (z1 + z2) / 2.0


# -----------------------------------------------------------------------------
# Analyzer (caricamento unico)
# -----------------------------------------------------------------------------

class Analyzer:
    def __init__(
        self,
        index_dir: Path,
        # Modelli
        backend: str = "clip",
        clip_model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        finetuned: Optional[str] = None,
        dino_model: str = "dinov2_vits14",
        # FAISS / ricerca
        search_backend: str = "auto",  # "auto" → usa FAISS se presente, altrimenti naive
        # SAM
        sam_ckpt: str = "weights/sam_vit_b_01ec64.pth",
        sam_model_type: str = "vit_b",
    ) -> None:
        # Backend di embedding
        self.backend_tuple = load_backend(
            backend=backend,
            clip_model=clip_model,
            pretrained=pretrained,
            dino_model=dino_model,
            finetuned=finetuned,
        )
        # Indice naive (torch)
        self.index_mat, self.meta = load_index(index_dir)
        # FAISS (opz.)
        self.use_faiss = (search_backend in ("auto", "faiss")) and faiss_exists(index_dir)
        self.faiss_idx = load_faiss_index(index_dir) if self.use_faiss else None
        # SAM generator (caricato una volta)
        self.sam_gen = load_sam_automatic(ckpt=sam_ckpt, model_type=sam_model_type)

    # --- singola immagine ---
    def analyze_one(
        self,
        scene_path: Path,
        out_dir: Path,
        # filtri/parametri scena
        min_area: int = 10000,
        max_masks: int = 12,
        min_area_ratio: float = 0.04,
        max_area_ratio: float = 0.55,
        expand_bbox_ratio: float = 0.18,
        bg_value: int = 0,
        topk: int = 5,
        keep_topk_objects: Optional[int] = None,
        prefer_diff_label: bool = False,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
        (out_dir / "json").mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(scene_path.as_posix())
        if bgr is None:
            raise FileNotFoundError(scene_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        crops_dir = out_dir / (scene_path.stem + "_crops")
        crops_dir.mkdir(parents=True, exist_ok=True)

        # SAM → maschere
        masks = generate_masks(bgr, self.sam_gen, min_area=min_area, max_masks=max_masks)

        preds_all = []
        overlay_cands = []

        for i, m in enumerate(masks, start=1):
            seg = m["segmentation"].astype(bool)
            x1, y1, x2, y2 = mask_to_bbox(seg)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            ratio = area / float(H * W)
            if ratio < float(min_area_ratio) or ratio > float(max_area_ratio):
                continue

            x1e, y1e, x2e, y2e = expand_bbox(x1, y1, x2, y2, H, W, ratio=expand_bbox_ratio)

            rgba = apply_mask_rgba(bgr, seg)
            bw, bh = x2e - x1e + 1, y2e - y1e + 1
            rel_pad = max(8, int(0.10 * min(bw, bh)))
            crop_rgba = crop_mask_tight(rgba, seg, pad=rel_pad)
            pil_rgb = rgba_to_rgb_for_clip(crop_rgba, bg_value=bg_value)

            z = embed_tta(pil_rgb, self.backend_tuple)

            if self.faiss_idx is not None:
                # FAISS top-1 (coseno con IP su vettori unitari)
                z_np = z.detach().cpu().numpy().astype(np.float32)
                scores, ids = search_faiss(self.faiss_idx, z_np, k=1)
                j = int(ids[0, 0]); score = float(scores[0, 0])
                label = self.meta[j]["label"]
            else:
                _vals, _idx = topk_cosine(z, self.index_mat, k=topk)
                label, score, _dbg = decide_label_baseline(z, self.index_mat, self.meta)

            preds_all.append((x1e, y1e, x2e, y2e, label, float(score)))

            crop_path = crops_dir / f"seg_{i:02d}_{label}_{score:.2f}.jpg"
            pil_rgb.save(crop_path.as_posix())
            overlay_cands.append((x1e, y1e, x2e, y2e, label, float(score)))

        overlay_nms = nms_preds(overlay_cands, iou_thr=0.30)
        overlay_final = overlay_nms
        if keep_topk_objects is not None:
            overlay_final = select_topK_distinct(
                overlay_nms,
                K=int(keep_topk_objects),
                score_backoff=(0.80, 0.70, 0.60),
                iou_thr=0.50,
                prefer_diff_label=prefer_diff_label,
                delta=0.10,
            )

        eval_json = {
            "scene": scene_path.name,
            "preds": [
                {"bbox": [int(x1), int(y1), int(x2), int(y2)], "label": lbl, "score": float(sc)}
                for (x1, y1, x2, y2, lbl, sc) in preds_all
            ],
        }
        (out_dir / f"json/{scene_path.stem}.json").write_text(
            json.dumps(eval_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        overlay_json = {
            "scene": scene_path.name,
            "preds": [
                {"bbox": [int(x1), int(y1), int(x2), int(y2)], "label": lbl, "score": float(sc)}
                for (x1, y1, x2, y2, lbl, sc) in overlay_final
            ],
        }
        (out_dir / f"json/{scene_path.stem}_overlay.json").write_text(
            json.dumps(overlay_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        overlay_path = out_dir / f"overlay/{scene_path.stem}_overlay.jpg"
        draw_overlay(rgb, overlay_final, overlay_path)

        return eval_json, overlay_json


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_scenes(root: Path) -> List[Path]:
    scenes = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    scenes.sort()
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch scene analysis con caricamento unico dei modelli")
    ap.add_argument("--scenes-dir", default="data/scenes_real/images", type=str)
    ap.add_argument("--index", default="index", type=str)
    ap.add_argument("--out", default="outputs_scenes", type=str)

    # Modelli/ricerca
    ap.add_argument("--backend", choices=["clip", "dino"], default="clip")
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--finetuned", default=None)
    ap.add_argument("--dino-model", default="dinov2_vits14")

    ap.add_argument("--search-backend", choices=["auto", "naive", "faiss"], default="auto")

    # SAM
    ap.add_argument("--sam-ckpt", default="weights/sam_vit_b_01ec64.pth")
    ap.add_argument("--sam-model-type", default="vit_b")

    # Filtri/euristiche scena
    ap.add_argument("--min-area", default=10000, type=int)
    ap.add_argument("--max-masks", default=12, type=int)
    ap.add_argument("--min-area-ratio", default=0.04, type=float)
    ap.add_argument("--max-area-ratio", default=0.55, type=float)
    ap.add_argument("--expand-bbox-ratio", type=float, default=0.18)
    ap.add_argument("--bg-value", type=int, default=0)
    ap.add_argument("--topk", default=5, type=int)
    ap.add_argument("--keep-topk-objects", type=int, default=None)
    ap.add_argument("--prefer-diff-label", action="store_true", default=False)

    # Batch behaviour
    ap.add_argument("--fail-fast", action="store_true", default=False)

    args = ap.parse_args()

    scenes = find_scenes(Path(args.scenes_dir))
    print(f"Trovate {len(scenes)} scene in {args.scenes_dir}")

    analyzer = Analyzer(
        index_dir=Path(args.index),
        backend=args.backend,
        clip_model=args.clip_model,
        pretrained=args.pretrained,
        finetuned=args.finetuned,
        dino_model=args.dino_model,
        search_backend=args.search_backend,
        sam_ckpt=args.sam_ckpt,
        sam_model_type=args.sam_model_type,
    )

    ok, failed = 0, 0
    out_root = Path(args.out)
    for scene_path in tqdm(scenes, desc="Analyzing scenes"):
        try:
            analyzer.analyze_one(
                scene_path=scene_path,
                out_dir=out_root,
                min_area=args.min_area,
                max_masks=args.max_masks,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                expand_bbox_ratio=args.expand_bbox_ratio,
                bg_value=args.bg_value,
                topk=args.topk,
                keep_topk_objects=args.keep_topk_objects,
                prefer_diff_label=args.prefer_diff_label,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[ERRORE] {scene_path}: {e}")
            if args.fail_fast:
                raise

    print(f"\nCompletato. OK={ok}  FAILED={failed}  (tot={len(scenes)})")


if __name__ == "__main__":
    main()
