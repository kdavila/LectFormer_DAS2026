
import cv2
import math
import numpy as np

from PIL import Image

class DatasetHelper:
    @staticmethod
    def rotate_image(image, rot_angle, scale, padding):
        # at any angle without losing pixels ...
        if scale > 1.0:
            # ...  final image will be bigger ... original diagonal might be too small to be safe in all cases
            # ... use diagonal of up-scaled image
            diag = math.sqrt(math.pow(image.shape[0] * scale, 2) + math.pow(image.shape[1] * scale, 2))
        else:
            # scale is 1.0 or smaller, using original diagonal will be safe
            diag = math.sqrt(math.pow(image.shape[0], 2) + math.pow(image.shape[1], 2))

        trans_x = int((diag - image.shape[1]) / 2)
        trans_y = int((diag - image.shape[0]) / 2)

        trans_s = round(int(diag))

        if len(image.shape) == 3:
            # RGB image
            translated = np.zeros((trans_s, trans_s, 3), np.uint8)
            translated[:, :, 0] = padding[0]
            translated[:, :, 1] = padding[1]
            translated[:, :, 2] = padding[2]
        else:
            # binary image
            translated = np.ones((trans_s, trans_s), np.uint8) * padding

        translated[trans_y:trans_y + image.shape[0], trans_x:trans_x + image.shape[1]] = image.copy()

        rot_angle = rot_angle % 360.0

        out_size = (translated.shape[1], translated.shape[0])

        center = (int(translated.shape[1] / 2.0), int(translated.shape[0] / 2.0))
        rot_mat = cv2.getRotationMatrix2D(center=center, angle=rot_angle, scale=scale)
        rotated = cv2.warpAffine(translated, rot_mat, out_size, borderValue=padding)

        return rotated

    @staticmethod
    def apply_scaling_rotation(size, image, rotation_value, zoom_value, padding_color):
        # apply rotation (if any)
        if rotation_value is not None:
            # check if zoom can / should be done at once  ...
            if zoom_value is not None:
                # use rotation transform to scale (faster)
                scale = zoom_value
            else:
                scale = 1.0

            image = DatasetHelper.rotate_image(image, rotation_value, scale, padding_color)
            post_cut = True
        elif zoom_value is not None:
            # no rotation, but zoom (scaling) is required ...
            target_size = (int(round(zoom_value * image.shape[1])), int(round(zoom_value * image.shape[0])))
            image = cv2.resize(image, target_size)

            # check if padding or cutting will be required ...
            if zoom_value < 1.0:
                # needs to be padded  ...
                w, h = size
                padding_w = w - image.shape[1]
                padding_h = h - image.shape[0]
                start_w = int(padding_w / 2)
                start_h = int(padding_h / 2)

                if len(image.shape) == 3:
                    # RGB padding ...
                    padded = np.zeros((h, w, 3), dtype=image.dtype)
                    padded[:, :, 0] = padding_color[0]
                    padded[:, :, 1] = padding_color[1]
                    padded[:, :, 2] = padding_color[2]
                else:
                    # Gray-scale padding
                    padded = np.ones((h, w), dtype=image.dtype) * padding_color

                padded[start_h:start_h + image.shape[0], start_w:start_w + image.shape[1]] = image

                image = padded

                post_cut = False
            else:
                # needs to be cut
                post_cut = True
        else:
            post_cut = False

        # crop or pad main region if rotation or zoom has been applied
        if post_cut:
            # horizontal
            w, h = size
            if w < image.shape[1]:
                # width of image is bigger than allowed ... cut ...
                start_w = int((image.shape[1] - w) / 2.0)
                image = image[:, start_w:start_w + w]

            # vertical
            if h < image.shape[0]:
                # height of image is bigger than allowed ... cut
                start_h = int((image.shape[0] - h) / 2.0)
                image = image[start_h:start_h + h, :]

        return image

    @staticmethod
    def npimg_safe_add_noise(raw_img, noise):
        raw_img = raw_img.astype(np.float64)
        raw_img += noise
        raw_img[raw_img < 0] = 0
        raw_img[raw_img > 255] = 255
        raw_img = raw_img.astype(np.uint8)

        return raw_img

    @staticmethod
    def pilimg_add_JPEG_compression_noise(pil_img, comp_value):
        comp_img = np.asarray(pil_img)
        flag, raw_data = cv2.imencode(".jpg", comp_img, params=(cv2.IMWRITE_JPEG_QUALITY, comp_value))
        comp_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(comp_img)

        return pil_img