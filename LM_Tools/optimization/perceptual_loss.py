
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

compile_disable_if_windows = (
    torch.compiler.disable
    if platform.system() == "Windows"
    else lambda f: f
)

class VGGHelper(nn.Module):
    def __init__(self, layers):
        super(VGGHelper, self).__init__()

        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg.eval()

        # Freeze VGG weights
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg_slices = nn.ModuleList()
        self.layer_name_mapping = {
            "relu1_1": 1, "relu1_2": 3,
            "relu2_1": 6, "relu2_2": 8,
            "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
            "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
            "relu5_1": 25
        }

        # Build slices (one module per requested layer)
        last_idx = 0
        for layer_name in layers:
            idx = self.layer_name_mapping[layer_name]
            self.vgg_slices.append(nn.Sequential(*self.vgg[last_idx:idx + 1]))
            last_idx = idx + 1

    @compile_disable_if_windows
    def run_vgg_slice(self, slice, x, y):
        return slice(x), slice(y)


# WARNING: this part was generated mainly by Chat-GPT
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3"), resize=False):
        super(VGGPerceptualLoss, self).__init__()

        self.vgg_helper = VGGHelper(layers)

        self.layers = layers
        self.resize = resize

        # VGG normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # both target and reconstruction are in the range -1, 1, but VGG expects a standard ImageNet normalization
        # [-1, +1] -> [0, 2] -> [0, 1]
        y = (y + 1.0) / 2.0
        x = (x + 1.0) / 2.0

        # Normalize to VGG convention
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        if self.resize:
            x = F.interpolate(x, mode="bilinear", size=(224, 224), align_corners=False)
            y = F.interpolate(y, mode="bilinear", size=(224, 224), align_corners=False)

        # compute the features per slice, and compute the local loss
        loss = 0.0
        for slice in self.vgg_helper.vgg_slices:
            x, y = self.vgg_helper.run_vgg_slice(slice, x, y)
            # x = slice(x)
            # y = slice(y)
            loss += torch.nn.functional.l1_loss(x, y)

        return loss


class PixWeightedVGGPerceptualLoss(nn.Module):
    def __init__(self, pos_weight, neg_weight, blur_size=7, layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3"), resize=False):
        super(PixWeightedVGGPerceptualLoss, self).__init__()

        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        self.vgg_helper = VGGHelper(layers)

        self.layers = layers
        self.resize = resize
        self.blur_size = blur_size

        # VGG normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y, mask):
        # both target and reconstruction are in the range -1, 1, but VGG expects a standard ImageNet normalization
        # [-1, +1] -> [0, 2] -> [0, 1]
        y = (y + 1.0) / 2.0
        x = (x + 1.0) / 2.0

        # Normalize to VGG convention
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        if self.resize:
            x = F.interpolate(x, mode="bilinear", size=(224, 224), align_corners=False)
            y = F.interpolate(y, mode="bilinear", size=(224, 224), align_corners=False)

        if self.blur_size is not None and self.blur_size > 0:
            blur_padding = (self.blur_size - 1) // 2
            mask = F.avg_pool2d(mask, kernel_size=self.blur_size, stride=1, padding=blur_padding)

        # compute the features per slice, and compute the local loss
        slice_outputs = []
        for slice in self.vgg_helper.vgg_slices:
            # x = slice(x)
            # y = slice(y)
            x, y = self.vgg_helper.run_vgg_slice(slice, x, y)
            slice_outputs.append((x, y))

        loss = 0.0
        # for slice in self.vgg_helper.vgg_slices:
        for x, y in slice_outputs:
            # x = slice(x)
            # y = slice(y)
            # x, y = self.vgg_helper.run_vgg_slice(slice, x, y)

            # Resize mask to current feature resolution
            m_res = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
            # (Optional) clamp to [0,1] in case of interpolation overshoot
            m_res = m_res.clamp(0.0, 1.0)
            # Broadcast mask across channels
            pix_weights = m_res.expand_as(x)

            # loss += torch.nn.functional.l1_loss(x, y)
            diff = (x - y).abs()
            diff = diff * (pix_weights * self.pos_weight + (1 - pix_weights) * self.neg_weight)

            loss = loss + diff.mean()

        return loss
