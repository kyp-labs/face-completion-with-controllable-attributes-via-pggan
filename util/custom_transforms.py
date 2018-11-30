"""Custom transforms for augmentation."""

import numbers
import random
import collections

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision.transforms import functional as F
from util.snapshot_transforms import PolygonMaskBase

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        tmp_img = sample['image']
        tmp_obs_mask = sample['obs_mask']
        tmp_real_mask = sample['real_mask']

        if random.random() < self.p:
            tmp_img = F.hflip(tmp_img)
            tmp_obs_mask = F.hflip(tmp_obs_mask)
            tmp_real_mask = F.hflip(tmp_real_mask)

        sample['image'] = tmp_img
        sample['obs_mask'] = tmp_obs_mask
        sample['real_mask'] = tmp_real_mask
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tmp = sample['image']
        sample['image'] = F.normalize(tmp, self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        tmp = sample['image']
        sample['image'] = F.resize(tmp, self.size, self.interpolation)
        return sample

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for elem in ['image', 'attr', 'real_mask', 'obs_mask']:
            if elem == 'attr':
                tmp = sample['attr']
                sample[elem] = torch.from_numpy(tmp).float().squeeze()

            else:
                tmp = sample[elem]
                sample[elem] = F.to_tensor(tmp)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        tmp = sample['image']
        sample['image'] = F.center_crop(tmp, self.size)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class PolygonMask(PolygonMaskBase):
    """Add Square mask to the sample."""

    def __init__(self, num_classes=1):
        """constructor."""
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        landmark = sample['landmark'] # tuple[10]
        gender = np.argmax(sample['attr'], axis=1)
        fake_gender = random.randint(0, 1)

        resolution = image.size[-1]
        real_mask = np.full([resolution, resolution], gender,
                            dtype=np.uint8)
        obs_mask = real_mask.copy()
        
        polygon_type = random.randint(0, 3)
        
        if polygon_type == 0:
            polygon_coords = self.make_face_mask(landmark, resolution)
        elif polygon_type == 1:
            polygon_coords = self.make_eye_mask(landmark, resolution)
        elif polygon_type == 2:
            polygon_coords = self.make_nose_mask(landmark, resolution)
        elif polygon_type == 3:
            polygon_coords = self.make_lip_mask(landmark, resolution)
        else:
            polygon_coords = self.make_face_mask(landmark, resolution)

        cv2.fillPoly(obs_mask, polygon_coords, fake_gender)
        
        sample['real_mask'] = Image.fromarray(np.int8(real_mask))
        sample['obs_mask'] = Image.fromarray(np.int8(obs_mask))
        sample['gender'] = int(gender)
        sample['fake_gender'] = fake_gender
        return sample

    def __repr__(self):  # noqa: D105
        return f'PolygonMask:(num_classes={str(self.num_classes)})'
