import sys
from pathlib import Path
from PIL import Image
import numpy as np


def convert_maskfg_to_binary(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_maskfg.png"))

    print(f"Found {len(files)} files.")

    for in_path in files:
        out_name = in_path.name.replace("_maskfg", "")
        arr = np.array(Image.open(in_path))

        # keep only class 100 (word)
        binary = np.where(arr == 100, 255, 0).astype(np.uint8)

        out_path = output_dir / out_name
        Image.fromarray(binary).save(out_path)

        print(f"Saved: {out_path}")

    print("Done.")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_dir out_dir")
        return

    in_dir = sys.argv[1]  # Path to TextSeg/semantic_label
    out_dir = sys.argv[2] # Path to TextSeg/binary_label

    convert_maskfg_to_binary(in_dir, out_dir)


if __name__ == "__main__":
    main()
