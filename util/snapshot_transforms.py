"""Custom transforms for augmentation."""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

class PermutePolygonMask(object):
    """Add Square mask to the sample."""

    def __init__(self, num_classes=1):
        """constructor."""
        self.num_classes = num_classes

 
    def calc_polygon(self, landmark, resolution):
        
        landmark_adjust_ratio = 256 // resolution
        
        polygon_coords = np.take(landmark,
                                 [0, 1, 2, 3, 2, 3, 8, 9, 6, 7, 0, 1])
        nose_coords = np.take(landmark, [4, 5])

        polygon_coords = polygon_coords.astype(np.int32).reshape(1, -1, 2)
        nose_coords = nose_coords.astype(np.int32).reshape(1, -1, 2)

        # eye mask width
        EYE_LEFT = 0
        EYE_RIGHT = 1
        eye_width = abs(polygon_coords[:, EYE_LEFT, 0]
                        - nose_coords[:, 0, 0]).astype(np.int32)
        polygon_coords[:, EYE_LEFT, 0] -= eye_width
        eye_width = abs(polygon_coords[:, EYE_RIGHT, 0]
                        - nose_coords[:, 0, 0]).astype(np.int32)
        polygon_coords[:, EYE_RIGHT, 0] += eye_width

        # eye mask height
        eye_height = abs(polygon_coords[:, EYE_LEFT, 1]
                         - nose_coords[:, 0, 1]).astype(np.int32)//2
        polygon_coords[:, EYE_LEFT, 1] -= eye_height
        eye_height = abs(polygon_coords[:, EYE_RIGHT, 1]
                         - nose_coords[:, 0, 1]).astype(np.int32)//2
        polygon_coords[:, EYE_RIGHT, 1] -= eye_height

        # cheek mask width
        CHEEK_LEFT = 5
        CHEEK_RIGHT = 2
        polygon_coords[:, CHEEK_LEFT, 0] = polygon_coords[:, EYE_LEFT, 0]
        polygon_coords[:, CHEEK_RIGHT, 0] = polygon_coords[:, EYE_RIGHT, 0]

        # cheek mask height
        polygon_coords[:, CHEEK_LEFT, 1] = nose_coords[:, 0, 1]
        polygon_coords[:, CHEEK_RIGHT, 1] = nose_coords[:, 0, 1]

        # lip mask width
        LIP_LEFT = 4
        LIP_RIGHT = 3

        # lip mask height
        lip_height = abs(polygon_coords[:, LIP_LEFT, 1]
                         - nose_coords[:, 0, 1]).astype(np.int32)//2
        polygon_coords[:, LIP_LEFT, 1] += lip_height
        lip_height = abs(polygon_coords[:, LIP_RIGHT, 1]
                         - nose_coords[:, 0, 1]).astype(np.int32)//2
        polygon_coords[:, LIP_RIGHT, 1] += lip_height

        polygon_coords = polygon_coords // landmark_adjust_ratio
        
        return polygon_coords
        
    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        landmark = sample['landmark']
        
        # calc polygon
        resolution = image.size[-1]
        polygon_coords = self.calc_polygon(landmark, resolution)
        # draw polygon on the real image
        masked_real = np.asarray(sample['image'].copy())

        thickness = 2
        cyan = (255, 255, 0)
        # magenta = (255, 0, 255)
        masked_real = cv2.polylines(masked_real,
                                    polygon_coords,
                                    True,
                                    cyan,
                                    thickness)
        sample['masked_real'] = Image.fromarray(masked_real)
        

        # make obs mask list
        obs_mask_list = []
        real_mask = np.full([resolution, resolution], 2, dtype=np.uint8)
        for attr in range(sample['attr'].shape[1]):
            obs_mask = real_mask.copy()
            cv2.fillPoly(obs_mask, polygon_coords, int(attr))
            obs_mask_list.append(Image.fromarray(np.int8(obs_mask)))

        sample['obs_mask_list'] = obs_mask_list
        return sample

    def __repr__(self):  # noqa: D105
        return f'PolygonMask:(num_classes={str(self.num_classes)})'
        
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
        tmp = sample['masked_real']
        sample['masked_real'] = F.normalize(tmp, self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + \
            '(mean={0}, std={1})'.format(self.mean, self.std)
    
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
        for elem in ['image', 'attr', 'masked_real', 'obs_mask_list']:
            if elem == 'attr':
                tmp = sample['attr']
                sample[elem] = torch.from_numpy(tmp).float().squeeze()

            elif elem == 'obs_mask_list':
                obs_mask_list = []
                for obs_mask in sample['obs_mask_list']:
                    obs_mask_list.append(F.to_tensor(obs_mask))
                sample[elem] = obs_mask_list
            else:
                tmp = sample[elem]
                sample[elem] = F.to_tensor(tmp)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'