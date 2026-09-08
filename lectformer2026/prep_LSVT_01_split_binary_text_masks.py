
import os
import sys
import random

import shutil

def copy_img_set(img_list, in_img_dir, in_mask_dir, out_img_dir, out_mask_dir):
    for idx, img_filename in enumerate(img_list):
        print(" " * 80, end="\r")
        print(f"Copying: {img_filename} ({idx + 1}/{len(img_list)})", end="\r")
        base, ext = os.path.splitext(img_filename)

        # copy the image
        src_path = f"{in_img_dir}/{img_filename}"
        dst_patch = f"{out_img_dir}/{img_filename}"
        shutil.copy(src_path, dst_patch)

        # copy the mask ...
        mask_filename = f"{base}.png"
        src_path = f"{in_mask_dir}/{mask_filename}"
        dst_patch = f"{out_mask_dir}/{mask_filename}"
        shutil.copy(src_path, dst_patch)
    print("")
    print("... finished!")


def main():
    if len(sys.argv) < 8:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_img_dir in_mask_dir prc_valid out_train_img_dir out_train_mask_dir out_valid_img_dir out_valid_mask_dir")
        print("With")
        print("\tin_img_dir\tPath to input image dir")
        print("\tin_mask_dir\tPath to input mask dir")
        print("\tprc_valid\tPercentage to be used as validation [0.0 to 1.0]")
        print("\tout_train_img_dir\tPath to output train image dir")
        print("\tout_train_mask_dir\tPath to output train mask dir")
        print("\tout_valid_img_dir\tPath to output valid image dir")
        print("\tout_valid_mask_dir\tPath to output valid mask dir")
        return

    in_img_dir = sys.argv[1]
    in_mask_dir = sys.argv[2]
    out_train_img_dir = sys.argv[4]
    out_train_mask_dir = sys.argv[5]
    out_valid_img_dir = sys.argv[6]
    out_valid_mask_dir = sys.argv[7]

    try:
        prc_valid = float(sys.argv[3])
        if prc_valid < 0.0 or prc_valid > 1.0:
            print("Invalid value for prc_valid. It must be between 0.0 and 1.0")
            return
    except:
        print("Invalid value for prc_valid. It must be a valid float")
        return

    in_images = os.listdir(in_img_dir)
    print(f"A total of {len(in_images)} images were found")

    n_training = int(round(len(in_images) * (1 - prc_valid)))
    n_validation = len(in_images) - n_training
    print(f" - {n_training} images will be used for training")
    print(f" - {n_validation} images will be used for validation")

    # shuffle (the lazy way)
    shuffled_images = sorted([(random.random(), img_filename) for img_filename in in_images])

    train_imgs = [img_filename for _, img_filename in shuffled_images[:n_training]]
    valid_imgs = [img_filename for _, img_filename in shuffled_images[n_training:]]

    print("Copying training split")
    copy_img_set(train_imgs, in_img_dir, in_mask_dir, out_train_img_dir, out_train_mask_dir)

    print("Copying validation split")
    copy_img_set(valid_imgs, in_img_dir, in_mask_dir, out_valid_img_dir, out_valid_mask_dir)

if __name__ == '__main__':
    main()
