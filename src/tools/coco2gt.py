# tools/coco2gt.py
import json, argparse
from pathlib import Path

def parse_map(s):
    # es: "1:Gara,2:Sakura,3:Naruto,4:Tsunade"
    out = {}
    for tok in s.split(","):
        k,v = tok.split(":")
        out[int(k.strip())] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO JSON (makesense export)")
    ap.add_argument("--out", required=True, help="Output gt.json")
    ap.add_argument("--catmap", default="", help="Override id->name (es. '1:Naruto,2:Gaara,3:Sakura,4:Tsunade')")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    # categories: prova a leggerle dal file; se mancano, richiedi --catmap
    id2name = {}
    if "categories" in coco:
        id2name = {c["id"]: c["name"] for c in coco["categories"]}
    if args.catmap:
        id2name.update(parse_map(args.catmap))
    if not id2name:
        raise SystemExit("Category names not found. Passa --catmap '1:Naruto,2:Gaara,3:Sakura,4:Tsunade'")

    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    # COCO bbox = [x,y,w,h] → [x1,y1,x2,y2]
    def to_xyxy(b):
        x,y,w,h = b
        return [int(x), int(y), int(x+w-1), int(y+h-1)]

    gt = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        file = id2file[img_id]
        lab = id2name[ann["category_id"]]
        x1,y1,x2,y2 = to_xyxy(ann["bbox"])
        gt.setdefault(file, []).append({"label": lab, "bbox": [x1,y1,x2,y2]})

    Path(args.out).write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {sum(len(v) for v in gt.values())} objects in {len(gt)} images.")

if __name__ == "__main__":
    main()
