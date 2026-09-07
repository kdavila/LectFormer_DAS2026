import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def process_split(split_dir: Path):
    img_dir = split_dir / "img"
    ann_dir = split_dir / "ann"
    mask_dir = split_dir / "text_masks"

    mask_dir.mkdir(exist_ok=True)

    json_files = sorted(ann_dir.glob("*.json"))

    for json_path in json_files:

        # example: img11.jpg.json -> img11.jpg
        img_name = json_path.name.replace(".json", "")
        img_path = img_dir / img_name

        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        # read image to get size
        img = Image.open(img_path)
        width, height = img.size

        mask = np.zeros((height, width), dtype=np.uint8)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        objects = data.get("objects", [])

        for obj in objects:

            if obj.get("classTitle") != "text":
                continue

            pts = obj["points"]["exterior"]

            if len(pts) < 3:
                continue

            poly = np.array(pts, dtype=np.int32)

            cv2.fillPoly(mask, [poly], 255)

        out_name = Path(img_name).stem + ".png"
        out_path = mask_dir / out_name

        Image.fromarray(mask).save(out_path)

        print("Saved:", out_path)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} totaltext_root_dir")
        return

    totaltext_root_dir = sys.argv[1]

    ROOT = Path(totaltext_root_dir)

    for split in ["train", "test"]:
        split_dir = ROOT / split
        print("Processing", split_dir)
        process_split(split_dir)


if __name__ == "__main__":
    main()
