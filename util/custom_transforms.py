"""Custom transforms for augmentation."""

import numbers
import random
import collections

import cv2
import numpy as np
import torch
from PIL import Image
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
        """Initialize."""
        self.p = p
        self.IMAGE_WIDTH = 256
        self.landmark_idx = {"left_eye" : [0,1],
                    "right_eye" : [2,3],
                    "nose" : [4,5],
                    "left_lip" : [6,7],
                    "right_lip" : [8,9]}
        
    def __call__(self, sample):
        """Call.

        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.

        """
        tmp_img = sample['image']
        tmp_landmark = sample['landmark']
        
        if random.random() < self.p:
            tmp_img = F.hflip(tmp_img)
            tmp_landmark = self.flip_landmark(tmp_landmark)
            
        sample['image'] = tmp_img
        sample['landmark'] = tmp_landmark
        return sample

    def flip_landmark(self, landmark):
        new_landmark = landmark.copy()
        for key in self.landmark_idx:
            idx = self.landmark_idx[key][0]
            new_landmark[idx] = self.IMAGE_WIDTH  - landmark[idx]
            
        return new_landmark
    
    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)``
    for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        """Initialize."""
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """Call.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.

        """
        tmp = sample['image']
        sample['image'] = F.normalize(tmp, self.mean, self.std)
        return sample

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + \
            '(mean={0}, std={1})'.format(self.mean, self.std)


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
        """Init."""
        assert isinstance(size, int) or \
            (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """Call.

        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.

        """
        tmp = sample['image']
        sample['image'] = F.resize(tmp, self.size, self.interpolation)
        return sample

    def __repr__(self):
        """Repr."""
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + \
            '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W)
    in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """Call.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.

        """
        for elem in ['image', 'attr', 'attr_obs', 'mask']:
            if elem in ['attr', 'attr_obs']:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp).float().squeeze()
            else:
                tmp = sample[elem]
                sample[elem] = F.to_tensor(tmp)
        return sample

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + '()'


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        """Init."""
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """Call.

        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.

        """
        tmp = sample['image']
        sample['image'] = F.center_crop(tmp, self.size)
        return sample

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class PolygonMask(PolygonMaskBase):
    """Add Square mask to the sample."""

    def __init__(self, attribute_size):
        """constructor."""
        super().__init__()
        self.attribute_size = attribute_size
        
    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        landmark = sample['landmark']  # tuple[10]
        attr_real = sample['attr']
        
        source_domain = int(np.argmax(attr_real, axis=1))
        target_domain = random.randint(0, self.attribute_size-1)
        
        N, attr_size = attr_real.shape
        attr_obs = np.zeros_like(attr_real)
        attr_obs[np.arange(N), target_domain] = 1
        
        resolution = image.size[-1]
        mask = np.full([resolution, resolution], source_domain,
                            dtype=np.uint8)

        polygon_type = random.randint(0, 3)
        polygon = self.get_polygon(polygon_type, landmark, resolution)


        cv2.fillPoly(mask, polygon, target_domain)

        sample['mask'] = Image.fromarray(np.int8(mask))
        sample['attr_obs'] = attr_obs
        sample['source_domain'] = source_domain
        sample['target_domain'] = target_domain
        return sample

    def get_polygon(self, polygon_type, landmark, resolution):
        """Get polygon."""
        if polygon_type == 0:
            polygon = self.make_face_mask(landmark, resolution)
        elif polygon_type == 1:
            polygon = self.make_eye_mask(landmark, resolution)
        elif polygon_type == 2:
            polygon = self.make_nose_mask(landmark, resolution)
        elif polygon_type == 3:
            polygon = self.make_lip_mask(landmark, resolution)
        else:
            polygon = self.make_face_mask(landmark, resolution)
 
        return polygon
    
    def __repr__(self):  # noqa: D105
        return f'PolygonMask:(num_classes={str(self.num_classes)})'
