import json
import shutil
import sys
from pathlib import Path


def export_split(split_name, image_dir, binary_label_dir, split_json_path, output_root, with_suffix=False):
    image_dir = Path(image_dir)
    binary_label_dir = Path(binary_label_dir)
    split_json_path = Path(split_json_path)
    output_root = Path(output_root)

    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split_name not in split_data:
        raise ValueError(
            f"Split '{split_name}' not found in {split_json_path}. "
            f"Available splits: {list(split_data.keys())}"
        )

    out_dir = output_root / split_name
    out_img_dir = out_dir / "img"
    out_mask_dir = out_dir / "masks"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    ids = split_data[split_name]

    missing_images = []
    missing_masks = []

    for data_id in ids:
        img_path = image_dir / f"{data_id}.jpg"
        if with_suffix:
            mask_path = binary_label_dir / f"{data_id}_maskfg.png"
        else:
            mask_path = binary_label_dir / f"{data_id}.png"

        if not img_path.exists():
            missing_images.append(str(img_path))
            continue

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        shutil.copy2(img_path, out_img_dir / img_path.name)
        shutil.copy2(mask_path, out_mask_dir / mask_path.name)

    print(f"Split: {split_name}")
    print(f"Requested IDs: {len(ids)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Output written to: {out_dir}")

    if missing_images:
        print("\nMissing images:")
        for p in missing_images[:20]:
            print(" ", p)
        if len(missing_images) > 20:
            print(f" ... and {len(missing_images) - 20} more")

    if missing_masks:
        print("\nMissing masks:")
        for p in missing_masks[:20]:
            print(" ", p)
        if len(missing_masks) > 20:
            print(f" ... and {len(missing_masks) - 20} more")


def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} image_dir bin_label_dir split_json out_root")
        return

    image_dir = sys.argv[1]             # path to TextSeg/image
    binary_label_dir = sys.argv[2]      # Path to exported (01) TextSeg/binary_label
    split_json_path = sys.argv[3]       # Path to TextSeg/split.json
    output_root = sys.argv[4]           # Output dir for splits
    in_has_suffix = True

    export_split("train", image_dir, binary_label_dir, split_json_path, output_root, with_suffix=in_has_suffix)
    export_split("val", image_dir, binary_label_dir, split_json_path, output_root, with_suffix=in_has_suffix)
    export_split("test", image_dir, binary_label_dir, split_json_path, output_root, with_suffix=in_has_suffix)

if __name__ == "__main__":
    main()
