"""
eval_scene.py

Scopo
-----
Valuta le predizioni di `analyze_scene` al livello oggetto (bbox + label):
- legge un ground truth per scena (`gt.json`) con lista di oggetti `{bbox,label}`;
- legge i file predetti per scena (modalità EVAL: `{scene}.json` oppure
  overlay: `{scene}_overlay.json` se `--use-overlay`);
- esegue un matching ottimo tra GT e pred tramite algoritmo di Hungarian
  (sul costo = 1−IoU, con soglia IoU minima);
- calcola accuracy globale top-1 e per-classe, più diagnostiche:
  - detection recall @IoU≥thr (quanti GT sono stati coperti da almeno una pred)
  - classification accuracy condizionata ai match accettati.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# Metriche di base
# -----------------------------------------------------------------------------

def iou_xyxy(a: List[int], b: List[int]) -> float:
    """Calcola l'IoU tra due bbox [x1,y1,x2,y2] (coordinate inclusive).

    Inclusività: la larghezza/altezza si calcola con `+1` per includere i bordi.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, x2 - x1 + 1), max(0, y2 - y1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    uni = area_a + area_b - inter
    return inter / uni if uni > 0 else 0.0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Valutazione object-level (top-1) con assegnamento Hungarian."
    )
    ap.add_argument("--gt", required=True, help="Path a gt.json {scene: [{bbox,label},...]}.")
    ap.add_argument(
        "--preds-dir",
        required=True,
        help="Cartella con {scene}.json (EVAL) o {scene}_overlay.json (se --use-overlay)",
    )
    ap.add_argument(
        "--use-overlay",
        action="store_true",
        help="Se presente, usa {scene}_overlay.json invece di {scene}.json.",
    )
    ap.add_argument("--iou-match", type=float, default=0.5, help="Soglia IoU minima per accettare un match.")
    ap.add_argument(
        "--label-map",
        default="",
        help="Mappa nomi pred→GT, es. 'Gara:Gaara;Sakura:Sakura'.",
    )
    ap.add_argument("--csv-out", default="", help="(Opzionale) Path a CSV riassuntivo per classe.")
    ap.add_argument("--verbose", action="store_true", help="Log dettagliato per scena.")
    args = ap.parse_args()

    # Pred→GT label normalization
    label_map: Dict[str, str] = {}
    if args.label_map:
        for tok in args.label_map.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            src, dst = tok.split(":")
            label_map[src.strip()] = dst.strip()

    gt_data = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    preds_dir = Path(args.preds_dir)

    total_gt = 0
    correct = 0

    per_cls_total = defaultdict(int)
    per_cls_correct = defaultdict(int)

    missing_preds = []

    # diagnostiche: detection & classification
    det_matched_sum = 0  # #GT coperti (almeno un match IoU≥thr)
    det_total_sum = 0    # #GT totali
    cls_corr_on_matched = 0  # #match con label corretta
    cls_matched_total = 0    # #match accettati

    for sname, gt_objs in gt_data.items():
        stem = Path(sname).stem
        pred_file = preds_dir / (f"{stem}_overlay.json" if args.use_overlay else f"{stem}.json")

        if not pred_file.exists():
            missing_preds.append(stem)
            if args.verbose:
                print(f"[WARN] Pred mancante per scena: {stem}")
            # I GT mancanti contano comunque nel denominatore
            total_gt += len(gt_objs)
            for g in gt_objs:
                per_cls_total[g["label"]] += 1
            continue

        pred_json = json.loads(pred_file.read_text(encoding="utf-8"))
        preds = pred_json.get("preds", [])

        # Normalizza label predette
        for p in preds:
            p["label"] = label_map.get(p["label"], p["label"])

        G = len(gt_objs)
        P = len(preds)

        # Denominatori
        total_gt += G
        for g in gt_objs:
            per_cls_total[g["label"]] += 1

        if G == 0 or P == 0:
            if args.verbose:
                print(f"[{stem}] G={G}, P={P} -> no match")
            continue

        # Matrice IoU (G×P)
        iou_mat = np.zeros((G, P), dtype=np.float32)
        for i, g in enumerate(gt_objs):
            gb = g["bbox"]
            for j, pr in enumerate(preds):
                iou_mat[i, j] = iou_xyxy(gb, pr["bbox"])

        thr = float(args.iou_match)
        valid = iou_mat >= thr
        if not valid.any():
            if args.verbose:
                print(f"[{stem}] Nessuna coppia con IoU≥{thr}.")
            continue

        # Hungarian sul costo (1−IoU), con invalidazioni sotto soglia
        cost = 1.0 - iou_mat
        cost[~valid] = 1e9  # grande costo per impedire match
        rows, cols = linear_sum_assignment(cost)

        # Conta corretti: solo match validi (IoU≥thr) con label esatta
        accepted = 0
        scene_correct = 0
        for i, j in zip(rows, cols):
            if iou_mat[i, j] < thr:
                continue
            accepted += 1
            gl = gt_objs[i]["label"]
            pl = preds[j]["label"]
            if pl == gl:
                correct += 1
                scene_correct += 1
                per_cls_correct[gl] += 1

        # diagnostica per aggregati
        det_matched_sum += min(accepted, len(gt_objs))
        det_total_sum += len(gt_objs)
        cls_corr_on_matched += scene_correct
        cls_matched_total += accepted

        if args.verbose:
            print(
                f"[{stem}] GT={G}  Pred={P}  Matched@IoU≥{thr}: {accepted}  Correct: {scene_correct}"
            )

    # Risultati complessivi
    acc = correct / total_gt if total_gt else 0.0
    print(f"\nObject-level Accuracy (top-1): {acc:.4f}  [{correct}/{total_gt}]")

    if per_cls_total:
        print("\nPer-class accuracy:")
        for cls in sorted(per_cls_total.keys()):
            t = per_cls_total[cls]
            c = per_cls_correct.get(cls, 0)
            a = c / t if t else 0.0
            print(f"  {cls:8s}: {a:.4f}  [{c}/{t}]")

    if missing_preds:
        print(f"\n[WARN] Mancano predizioni per {len(missing_preds)} scene presenti nel GT:")
        for n in missing_preds[:10]:
            print("   -", n)
        if len(missing_preds) > 10:
            print("   ...")

    # Diagnostiche utili
    if det_total_sum > 0:
        det_recall = det_matched_sum / det_total_sum
        print(
            f"\n[Diag] Detection recall @IoU≥{args.iou_match}: {det_recall:.4f}  [{det_matched_sum}/{det_total_sum}]"
        )
    if cls_matched_total > 0:
        cls_acc_cond = cls_corr_on_matched / cls_matched_total
        print(
            f"[Diag] Classification accuracy | matched: {cls_acc_cond:.4f}  [{cls_corr_on_matched}/{cls_matched_total}]"
        )

    # CSV opzionale per report
    if args.csv_out:
        import csv

        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "correct", "total", "accuracy"])
            for cls in sorted(per_cls_total.keys()):
                t = per_cls_total[cls]
                c = per_cls_correct.get(cls, 0)
                a = c / t if t else 0.0
                w.writerow([cls, c, t, f"{a:.4f}"])
        print(f"\nSalvato CSV per-classe in: {args.csv_out}")


if __name__ == "__main__":
    main()
