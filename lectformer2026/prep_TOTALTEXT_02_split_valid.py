import shutil
import random
import sys
from pathlib import Path


def copy_set(ann_dir, mask_dir, file_set, out_root):
    for img_path in file_set:
        stem = img_path.stem  # imgXX

        ann_path = ann_dir / f"{img_path.name}.json"
        mask_path = mask_dir / f"{stem}.png"

        # copy img
        shutil.copy2(img_path, out_root / "img" / img_path.name)

        # copy ann
        if ann_path.exists():
            shutil.copy2(ann_path, out_root / "ann" / ann_path.name)

        # copy mask
        if mask_path.exists():
            shutil.copy2(mask_path, out_root / "masks" / mask_path.name)

def split_dataset(dataset_path, val_ratio=0.1, seed=1234):

    dataset_path = Path(dataset_path)

    img_dir = dataset_path / "img"
    ann_dir = dataset_path / "ann"
    mask_dir = dataset_path / "masks"

    assert img_dir.exists()
    assert ann_dir.exists()
    assert mask_dir.exists()

    parent = dataset_path.parent
    name = dataset_path.name

    train_out = parent / f"{name}_sub_train"
    valid_out = parent / f"{name}_sub_valid"

    for out_dir in [train_out, valid_out]:
        for sub in ["img", "ann", "masks"]:
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # list images
    img_files = sorted(img_dir.glob("*.jpg"))

    random.seed(seed)
    random.shuffle(img_files)

    n_total = len(img_files)
    n_val = int(n_total * val_ratio)

    val_files = set(img_files[:n_val])
    train_files = set(img_files[n_val:])

    print(f"Total: {n_total}")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(val_files)}")

    copy_set(ann_dir, mask_dir, train_files, train_out)
    copy_set(ann_dir, mask_dir, val_files, valid_out)

    print("Done")


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print(f"\tpython {sys.argv[0]} path_to_train")
        return

    path_to_train = sys.argv[1]     # path to total-text/train

    split_dataset(path_to_train, val_ratio=0.1, seed=42)


if __name__ == "__main__":
    main()
