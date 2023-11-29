"""
Augmentations for object detection.
TODO: Use albumentations or torchvision transforms v2.
References: most functions are taken from torchvision references.
(https://github.com/pytorch/vision/blob/main/references/detection/transforms.py)
"""
import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        image = F.pil_to_tensor(image)
        return image, target


class Normalize(nn.Module):
    def __init__(
        self,
        mean: list[float] | tuple[float, ...] | Tensor,
        std: list[float] | tuple[float, ...] | Tensor,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        image = image.float()
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, target


class Resize(nn.Module):
    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        image = F.resize(image, [self.size, self.size], self.interpolation)
        if target is not None:
            height, width = F.get_image_size(image)
            scale_x = width / self.size
            scale_y = height / self.size
            target["boxes"][:, [0, 2]] *= scale_x
            target["boxes"][:, [1, 3]] *= scale_y
            if "masks" in target:
                raise NotImplementedError
            if "keypoints" in target:
                target["keypoints"][:, :, 0] *= scale_x
                target["keypoints"][:, :, 1] *= scale_y
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target
