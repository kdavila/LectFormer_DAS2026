
from ..configuration.configuration import Configuration
from numpy.random import default_rng

class DataAugmentationParameters:
    def __init__(self, flip_chance=None, color_change_chance=None, color_invert_chance=None,
                 luminosity_changes_chance=None, gaussian_noise_chance=None, gaussian_noise_range=None,
                 zoom_chance=None, zoom_min=None, zoom_max=None, rotation_chance=None, rotation_range=None,
                 compress_chance=None, compress_min_q=None, compress_max_q=None):
        self.flip_chance = flip_chance

        self.color_change_chance = color_change_chance
        self.color_invert_chance = color_invert_chance
        self.luminosity_changes_chance = luminosity_changes_chance

        self.gaussian_noise_chance = gaussian_noise_chance
        self.gaussian_noise_range = gaussian_noise_range

        self.zoom_chance = zoom_chance
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max

        self.rotation_chance = rotation_chance
        self.rotation_range = rotation_range

        self.compression_noise_chance = compress_chance
        self.compression_noise_min_q = compress_min_q
        self.compression_noise_max_q = compress_max_q

        self.rng = default_rng()

        # other values less commonly adjusted ...
        # Luminosity default: from 0.75 - 1.0 with 50%, and 1.0-1.5 with 50%
        self.luminosity_min = 0.75  # must be below 1.0
        self.luminosity_max = 1.50  # must be above 1.0
        # -0.45 to 0.45 radians of the original hue value ...
        self.hue_range = 0.45
        # Contrast default: 0.50-1.0 with 50%, 1.0-2.0 with 50%
        self.contrast_min = 0.5
        self.contrast_max = 2.0
        # Gamma default: 0.50-1.0 with 50%, 1.0-2.0 with 50%
        self.gamma_min = 0.5
        self.gamma_max = 2.0
        # Saturation default: 0.25 - 1.0 with 50%, 1.0 - 5.0 with 50%
        self.saturation_min = 0.25
        self.saturation_max = 5.0

    def __str__(self):
        return (
            f"Data Augmentation Parameters:\n"
            f"  flip_chance: {self.flip_chance}\n"
            f"\n"
            f"  Color:\n"
            f"    color_change_chance: {self.color_change_chance}\n"
            f"    color_invert_chance: {self.color_invert_chance}\n"
            f"    luminosity_changes_chance: {self.luminosity_changes_chance}\n"
            f"    luminosity_min: {self.luminosity_min}\n"
            f"    luminosity_max: {self.luminosity_max}\n"
            f"    hue_range: {self.hue_range}\n"
            f"    contrast_min: {self.contrast_min}\n"
            f"    contrast_max: {self.contrast_max}\n"
            f"    gamma_min: {self.gamma_min}\n"
            f"    gamma_max: {self.gamma_max}\n"
            f"    saturation_min: {self.saturation_min}\n"
            f"    saturation_max: {self.saturation_max}\n"
            f"\n"
            f"  Gaussian Noise:\n"
            f"    gaussian_noise_chance: {self.gaussian_noise_chance}\n"
            f"    gaussian_noise_range: {self.gaussian_noise_range}\n"
            f"\n"
            f"  Zoom:\n"
            f"    zoom_chance: {self.zoom_chance}\n"
            f"    zoom_min: {self.zoom_min}\n"
            f"    zoom_max: {self.zoom_max}\n"
            f"\n"
            f"  Rotation:\n"
            f"    rotation_chance: {self.rotation_chance}\n"
            f"    rotation_range: {self.rotation_range}\n"
            f"\n"
            f"  Compression Noise:\n"
            f"    compression_noise_chance: {self.compression_noise_chance}\n"
            f"    compression_noise_min_q: {self.compression_noise_min_q}\n"
            f"    compression_noise_max_q: {self.compression_noise_max_q}"
        )

    def random_zoom_and_rotation_values(self):
        if self.zoom_chance is not None and self.rng.random() < self.zoom_chance:
            # randomly do a zoom ... decide if it will be a zoom in or zoom out
            if self.rng.random() < 0.5:
                # do a zoom out
                zoom_value = self.zoom_min + (1.0 - self.zoom_min) * self.rng.random()
            else:
                # do a zoom in
                zoom_value = 1.0 + (self.zoom_max - 1.0) * self.rng.random()
        else:
            zoom_value = None

        if self.rotation_chance is not None and self.rng.random() < self.rotation_chance:
            # do a random rotation
            rotation_value = -self.rotation_range + (2.0 * self.rotation_range * self.rng.random())
        else:
            rotation_value = None

        return zoom_value, rotation_value

    def should_apply_flip(self):
        return self.flip_chance is not None and self.rng.random() < self.flip_chance

    def should_apply_color_change(self):
        return self.color_change_chance is not None and self.rng.random() < self.color_change_chance

    def should_apply_color_invert(self):
        return self.color_invert_chance is not None and self.rng.random() < self.color_invert_chance

    def should_apply_luminosity_changes(self):
        return self.luminosity_changes_chance is not None and self.rng.random() < self.luminosity_changes_chance

    def should_apply_gaussian_noise(self):
        return self.gaussian_noise_chance is not None and self.rng.random() < self.gaussian_noise_chance

    def should_apply_compression_noise(self):
        return self.compression_noise_chance is not None and self.rng.random() < self.compression_noise_chance

    def get_TF_hue_value(self):
        return self.rng.random() * (self.hue_range * 2) - self.hue_range

    def get_TF_luminosity_value(self):
        # Apply random changes that affect the luminosity of the image
        if self.rng.standard_normal(1)[0] < 0:
            # lower brightness ... uniform ... from MIN to 1.0
            aug_val = 1.0 - self.rng.random() * (1.0 - self.luminosity_min)
        else:
            # increase brightness ... uniform ... from 1.0 to MAX
            aug_val = 1.0 + self.rng.random() * (self.luminosity_max - 1.0)
        return aug_val

    def get_TF_contrast_value(self):
        if self.rng.standard_normal(1)[0] < 0:
            # lower contrast ... uniform ... from MIN to 1.0
            aug_val = 1.0 - self.rng.random() * (1.0 - self.contrast_min)
        else:
            # increase contrast ... uniform ... from 1.0 to 2.0
            aug_val = 1.0 + self.rng.random() * (self.contrast_max - 1.0)
        return aug_val

    def get_TF_gamma_value(self):
        if self.rng.standard_normal(1)[0] < 0:
            # lower gamma ... uniform ... from 0.50 to 1.0
            aug_val = 1.0 - self.rng.random() * (1.0 - self.gamma_min)
        else:
            # increase gamma ... uniform ... from 1.0 to 2.0
            aug_val = 1.0 + self.rng.random() * (self.gamma_max - 1.0)
        return aug_val

    def get_TF_saturation_value(self):
        if self.rng.standard_normal(1)[0] < 0:
            # lower the saturation ... uniform ... between 0.25 to 1.0 saturation
            aug_val = 1.0 - self.rng.random() * (1.0 - self.saturation_min)
        else:
            # increase the saturation ... uniform ... between 1.0 to 5.0
            aug_val = 1.0 + self.rng.random() * (self.saturation_max - 1.0)
        return aug_val

    def get_CV_compression_value(self):
        comp_range = self.compression_noise_max_q - self.compression_noise_min_q
        return int(round(self.compression_noise_min_q + self.rng.random() * comp_range))

    def get_random(self):
        return self.rng.random()

    @staticmethod
    def FromConfiguration(augmentation_config):
        assert isinstance(augmentation_config, Configuration)

        flip_chance = augmentation_config.get("Flip.Chance", 0.5)
        color_change_chance = augmentation_config.get("ColorChange.Chance", 0.5)
        color_invert_chance = augmentation_config.get("ColorInvert.Chance", None)

        lum_change_chance = augmentation_config.get("LuminosityChange.Chance", 0.5)
        noise_chance = augmentation_config.get("GaussianNoise.Chance", 0.25)
        noise_level = augmentation_config.get("GaussianNoise.Level", 15.0)

        zoom_chance = augmentation_config.get("Zoom.Chance", 0.5)
        zoom_min_val = augmentation_config.get("Zoom.Min", 0.8)
        zoom_max_val = augmentation_config.get("Zoom.Max", 1.25)

        rotation_chance = augmentation_config.get("Rotation.Chance", 0.5)
        rotation_max = augmentation_config.get("Rotation.Max", 15)

        compress_chance = augmentation_config.get("Compression.Chance", 0.50)
        compress_min_q = augmentation_config.get("Compression.MinQuality", 0.25)
        compress_max_q = augmentation_config.get("Compression.MaxQuality", 0.75)

        return DataAugmentationParameters(flip_chance, color_change_chance, color_invert_chance, lum_change_chance,
                                          noise_chance, noise_level, zoom_chance, zoom_min_val, zoom_max_val,
                                          rotation_chance, rotation_max, compress_chance, compress_min_q,
                                          compress_max_q)

