"""Custom transforms for augmentation."""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

X = 0
Y = 1


class PolygonMaskBase(object):
    """Make Polygon Area of face, eyes, nose and lips using landmark."""

    def __init__(self):
        """Initilze."""
        self.landmark_idx = {"left_eye": [0, 1],
                             "right_eye": [2, 3],
                             "nose": [4, 5],
                             "left_lip": [6, 7],
                             "right_lip": [8, 9]}

    def convert_landmark_coord(self, landmark):
        """Convert landmark corrdinate."""
        # Landmark Key Points
        self.left_eye = np.take(landmark, self.landmark_idx["left_eye"])
        self.right_eye = np.take(landmark, self.landmark_idx["right_eye"])
        self.nose = np.take(landmark, self.landmark_idx["nose"])
        self.left_lip = np.take(landmark, self.landmark_idx["left_lip"])
        self.right_lip = np.take(landmark, self.landmark_idx["right_lip"])

    def make_face_mask(self, landmark, resolution):
        """Make face mask."""
        self.convert_landmark_coord(landmark)

        # Polygon Coordinae Intiailziation
        LEYE, REYE, RCHEEK, RLIP, LLIP, LCHEEK = 0, 1, 2, 3, 4, 5
        num_point = 6
        polygon = np.zeros((num_point, 2), dtype=np.int32)
        # polygon = polygon.astype(np.int32).reshape(1, -1, 2)

        # left eye X
        eye_width = self.left_eye[X] - self.nose[X]
        eye_width = abs(eye_width).astype(np.int32)
        polygon[LEYE, X] = self.left_eye[X] - eye_width

        # left eye Y
        eye_height = self.left_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[LEYE, Y] = self.left_eye[Y] - eye_height

        # right eye X
        eye_width = self.right_eye[X] - self.nose[X]
        eye_width = abs(eye_width).astype(np.int32)
        polygon[REYE, X] = self.right_eye[X] + eye_width

        # right eye Y
        eye_height = self.right_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[REYE, Y] = self.right_eye[Y] - eye_height

        # left cheek X
        polygon[LCHEEK, X] = polygon[LEYE, X]

        # left cheek Y
        polygon[LCHEEK, Y] = self.nose[1]

        # right cheek X
        polygon[RCHEEK, X] = polygon[REYE, X]

        # right cheek y
        polygon[RCHEEK, Y] = self.nose[1]

        # left lip X
        polygon[LLIP, X] = self.left_lip[X]

        # left lip Y
        lip_height = self.left_lip[Y] - self.nose[Y]
        lip_height = abs(lip_height).astype(np.int32) * 2 // 3
        polygon[LLIP, Y] = self.left_lip[Y] + lip_height

        # right lip X
        polygon[RLIP, X] = self.right_lip[X]

        # right lip Y
        lip_height = abs(self.right_lip[Y] - self.nose[Y])
        lip_height = lip_height.astype(np.int32) * 2 // 3
        polygon[RLIP, Y] = self.right_lip[Y] + lip_height

        polygon = polygon.reshape((1, -1, 2))

        landmark_adjust_ratio = 256 // resolution
        polygon = polygon // landmark_adjust_ratio

        return polygon

    def make_eye_mask(self, landmark, resolution):
        """Make eye mask."""
        self.convert_landmark_coord(landmark)

        ULEYE, UREYE, LREYE, LLEYE = 0, 1, 2, 3
        num_point = 4
        polygon = np.zeros((num_point, 2), dtype=np.int32)

        # upper left eye X
        eye_width = self.left_eye[X] - self.nose[X]
        eye_width = abs(eye_width).astype(np.int32)
        polygon[ULEYE, X] = self.left_eye[X] - eye_width

        # upper left eye Y
        eye_height = self.left_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[ULEYE, Y] = self.left_eye[Y] - eye_height

        # upper right eye X
        eye_width = self.right_eye[X] - self.nose[X]
        eye_width = abs(eye_width).astype(np.int32)
        polygon[UREYE, X] = self.right_eye[X] + eye_width

        # upper right eye Y
        eye_height = self.right_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[UREYE, Y] = self.right_eye[Y] - eye_height

        # lower left eye X
        polygon[LLEYE, X] = polygon[ULEYE, X]

        # lower left eye Y
        eye_height = self.left_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[LLEYE, Y] = self.left_eye[Y] + eye_height

        # lower right eye X
        polygon[LREYE, X] = polygon[UREYE, X]

        # lower right eye Y
        eye_height = self.right_eye[Y] - self.nose[Y]
        eye_height = abs(eye_height).astype(np.int32) / 2
        polygon[LREYE, Y] = self.right_eye[Y] + eye_height

        polygon = polygon.reshape((1, -1, 2))

        landmark_adjust_ratio = 256 // resolution
        polygon = polygon // landmark_adjust_ratio

        return polygon

    def make_nose_mask(self, landmark, resolution):
        """Make nose mask."""
        self.convert_landmark_coord(landmark)

        ULNOSE, URNOSE, LRNOSE, LLNOSE = 0, 1, 2, 3
        num_point = 4
        polygon = np.zeros((num_point, 2), dtype=np.int32)

        # upper left nose X
        nose_width = self.left_eye[X] - self.nose[X]
        nose_width = abs(nose_width).astype(np.int32) / 2
        polygon[ULNOSE, X] = self.nose[X] - nose_width

        # upper left nose Y
        polygon[ULNOSE, Y] = self.left_eye[Y]

        # upper right nose X
        nose_width = self.right_eye[X] - self.nose[X]
        nose_width = abs(nose_width).astype(np.int32) / 2
        polygon[URNOSE, X] = self.nose[X] + nose_width

        # upper right nose Y
        polygon[URNOSE, Y] = self.right_eye[Y]

        # lower left nose X
        polygon[LLNOSE, X] = polygon[ULNOSE, X]

        # lower left nose Y
        nose_height = self.left_lip[Y] - self.nose[Y]
        nose_height = abs(nose_height).astype(np.int32) / 2
        polygon[LLNOSE, Y] = self.nose[Y] + nose_height

        # lower right nose X
        polygon[LRNOSE, X] = polygon[URNOSE, X]

        # lower right nose Y
        nose_height = self.right_lip[Y] - self.nose[Y]
        nose_height = abs(nose_height).astype(np.int32) / 2
        polygon[LRNOSE, Y] = self.nose[Y] + nose_height

        polygon = polygon.reshape((1, -1, 2))

        landmark_adjust_ratio = 256 // resolution
        polygon = polygon // landmark_adjust_ratio

        return polygon

    def make_lip_mask(self, landmark, resolution):
        """Make lip mask."""
        self.convert_landmark_coord(landmark)

        ULLIP, URLIP, LRLIP, LLLIP = 0, 1, 2, 3
        num_point = 4
        polygon = np.zeros((num_point, 2), dtype=np.int32)
        offset = 3
        # upper left lip X
        polygon[ULLIP, X] = self.left_lip[X] - offset

        # upper left lip Y
        lip_height = self.left_lip[Y] - self.nose[Y]
        lip_height = abs(lip_height).astype(np.int32) / 2
        polygon[ULLIP, Y] = self.left_lip[Y] - lip_height

        # upper right lip X
        polygon[URLIP, X] = self.right_lip[X] + offset

        # upper right lip Y
        lip_height = self.right_lip[Y] - self.nose[Y]
        lip_height = abs(lip_height).astype(np.int32) / 2
        polygon[URLIP, Y] = self.right_lip[Y] - lip_height

        # lower left lip X
        polygon[LLLIP, X] = self.left_lip[X] - offset

        # lower left lip Y
        lip_height = self.left_lip[Y] - self.nose[Y]
        lip_height = abs(lip_height).astype(np.int32) * 2 // 3
        polygon[LLLIP, Y] = self.left_lip[Y] + lip_height

        # lower right lip X
        polygon[LRLIP, X] = self.right_lip[X] + offset

        # lower right lip Y
        lip_height = self.right_lip[Y] - self.nose[Y]
        lip_height = abs(lip_height).astype(np.int32) * 2 // 3
        polygon[LRLIP, Y] = self.right_lip[Y] + lip_height

        polygon = polygon.reshape((1, -1, 2))

        landmark_adjust_ratio = 256 // resolution
        polygon = polygon // landmark_adjust_ratio

        return polygon


class PermutePolygonMask(PolygonMaskBase):
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
        landmark = sample['landmark']

        # calc polygon
        resolution = image.size[-1]
        polygon1 = self.make_face_mask(landmark, resolution)
        polygon2 = self.make_eye_mask(landmark, resolution)
        polygon3 = self.make_nose_mask(landmark, resolution)
        polygon4 = self.make_lip_mask(landmark, resolution)
        polygon_list = [polygon1, polygon2, polygon3, polygon4]

        obs_mask_list = []
        masked_real_list = []
        real_mask = np.full([resolution, resolution], 2, dtype=np.uint8)
        for polygon in polygon_list:
            # draw polygon on the real image
            masked_real = np.asarray(sample['image'].copy())

            thickness = 2
            cyan = (255, 255, 0)
            masked_real = cv2.polylines(masked_real,
                                        polygon,
                                        True,
                                        cyan,
                                        thickness)

            masked_real_list.append(Image.fromarray(masked_real))

            # make obs mask list
            sub_obs_mask_list = []
            for attr in range(sample['attr'].shape[1]):
                obs_mask = real_mask.copy()
                cv2.fillPoly(obs_mask, polygon, int(attr))
                sub_obs_mask_list.append(Image.fromarray(np.int8(obs_mask)))

            obs_mask_list.append(sub_obs_mask_list)

        sample['masked_real_list'] = masked_real_list
        sample['obs_mask_list'] = obs_mask_list
        return sample

    def __repr__(self):  # noqa: D105
        """Repr."""
        return f'PolygonMask:(num_classes={str(self.num_classes)})'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

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
        masked_real_list = []
        for masked_real in sample['masked_real_list']:
            masked_real = F.normalize(masked_real, self.mean, self.std)
            masked_real_list.append(masked_real)
        sample['masked_real_list'] = masked_real_list

        return sample

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + \
            '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    """

    def __call__(self, sample):
        """Call.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.

        """
        for elem in ['image', 'attr', 'masked_real_list', 'obs_mask_list']:
            if elem == 'attr':
                tmp = sample['attr']
                sample[elem] = torch.from_numpy(tmp).float().squeeze()

            elif elem == 'obs_mask_list':
                obs_mask_list = []
                for sub_obs_mask_list in sample['obs_mask_list']:
                    sub_list = []
                    for obs_mask in sub_obs_mask_list:
                        sub_list.append(F.to_tensor(obs_mask))
                    obs_mask_list.append(sub_list)
                sample[elem] = obs_mask_list
            elif elem == 'masked_real_list':
                masked_real_list = []
                for masked_real in sample['masked_real_list']:
                    masked_real_list.append(F.to_tensor(masked_real))
                sample[elem] = masked_real_list
            else:
                tmp = sample[elem]
                sample[elem] = F.to_tensor(tmp)
        return sample

    def __repr__(self):
        """Repr."""
        return self.__class__.__name__ + '()'
