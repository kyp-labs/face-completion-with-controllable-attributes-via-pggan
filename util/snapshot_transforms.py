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
        """Make face mask.
        
        Args:
            landmark (tuple(8)) : face landmark
            resolution : target resolution for scaling landmark

        Returns:
            polygon (ndarray): face mask polygon

        """   
        
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
        """Make eye mask.
        
        Args:
            landmark (tuple(8)) : face landmark
            resolution : target resolution for scaling landmark

        Returns:
            polygon (ndarray): eye mask polygon

        """         
        self.convert_landmark_coord(landmark)

        ULEYE, UREYE, LREYE, LLEYE = 0, 1, 2, 3
        num_point = 4
        polygon = np.zeros((num_point, 2), dtype=np.int32)

        if self.left_eye[X] > self.right_eye[X]:
            print("left eye is greater thann right eye")
            
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
        """Make nose mask.
        
        Args:
            landmark (tuple(8)) : face landmark
            resolution : target resolution for scaling landmark

        Returns:
            polygon (ndarray): nose mask polygon

        """  
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
        """Make lip mask.
        
        Args:
            landmark (tuple(8)) : face landmark
            resolution : target resolution for scaling landmark

        Returns:
            polygon (ndarray): lip mask polygon

        """  
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
    
    def get_polygon(self, polygon_type, landmark, resolution):
        """Get Polygon Mask.
        
        Args:
            polygon_type (int): [0,3]
            landmark (tuple(8)) : face landmark
            resolution : target resolution for scaling landmark

        Returns:
            polygon (ndarray): mask polygon

        """        
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

class PermutePolygonMask(PolygonMaskBase):
    """Add Polygon mask to the sample."""
           
    def __init__(self, attribute_size, augmented_domain, mask_type_list):
        """Constructor.
        
        Args:
            attribute_size (int) : number of domains
            augmented_domain (ndarray) : [source, target, mask] domain number
            mask_type_list (list) : string mask type
                                    ["face", "eye", "nose","lip"]

        """
        super().__init__()
        self.augmented_domain = augmented_domain
        self.attribute_size = attribute_size
        self.mask_type_list = mask_type_list
        self.num_mask = len(self.mask_type_list)
        
    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        landmark = sample['landmark']
        source_domain = int(np.argmax(sample['attr'], axis=1))

        target_domain_set = self.get_target_domain(source_domain)
        
        # calc polygon
        resolution = image.size[-1]
        
        CONTEXT_BIT = 0
        MASK_BIT = 1
        
        polygon1 = self.make_face_mask(landmark, resolution)
        # polygon2 = self.make_eye_mask(landmark, resolution)
        # polygon3 = self.make_nose_mask(landmark, resolution)
        # polygon4 = self.make_lip_mask(landmark, resolution)
        # polygon_list = [polygon1, polygon2, polygon3, polygon4]
        polygon_list = [polygon1]
        
        mask_list = []
        masked_real_list = []
        mask_bits = np.full([resolution, resolution],
                            CONTEXT_BIT,
                            dtype=np.uint8)
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
            sub_mask_list = []
            for target_domain in target_domain_set:
                mask = mask_bits.copy()
                cv2.fillPoly(mask, polygon, MASK_BIT)
                sub_mask_list.append(Image.fromarray(np.int8(mask)))
                
            mask_list.append(sub_mask_list)

        sample['masked_real_list'] = masked_real_list
        sample['mask_list'] = mask_list
        sample['source_domain'] = source_domain
        sample['target_domain_list'] = target_domain_set
        return sample

    def get_target_domain(self, source_domain):
        """Make agumented domain one hot vector

        Args:
            source_domain (tensor) : [batch_size]
                                     source domain id
        """
        target_domain = self.augmented_domain[source_domain]
        target_domain = target_domain.flatten()
        target_domain = [x for x in target_domain if x != source_domain]
        target_domain.append(source_domain)
        return target_domain

    def __repr__(self):  # noqa: D105
        return f'PolygonMask:(num_classes={str(self.num_classes)})'
  
class PermuteDomainPolygonMask(PolygonMaskBase):
    """Add Polygon mask set to the sample."""
           
    def __init__(self,
                 attribute_size,
                 augmented_domain,
                 domain_lookup,
                 mask_type_list):
        """Constructor.
        
        Args:
            attribute_size (int) : number of domains
            augmented_domain (ndarray) : [source, target, mask] domain number
            domain_lookup (ndarray) : [doamin] [source, target, mask]
            mask_type_list (list) : string mask type
                                    ["face", "eye", "nose","lip"]

        """
        super().__init__()
        self.augmented_domain = augmented_domain
        self.domain_lookup = domain_lookup
        self.attribute_size = attribute_size
        self.mask_type_list = mask_type_list
        self.num_mask = len(self.mask_type_list)
        
    def __call__(self, sample):
        """caller.

        Args:
            sample (dict): {str: array} formatted data for training.

        Returns:
            sample (dict): {str: array} formatted data for training.

        """
        image = sample['image']
        landmark = sample['landmark']
        source_domain = int(np.argmax(sample['attr'], axis=1))

        target_domain_set = self.get_target_domain(source_domain)
        resolution = image.size[-1]
        
        CONTEXT_BIT = 0
        MASK_BIT = 1
        
        # calc polygon
        polygon_list = []
        for polygon_type in range(self.num_mask):
            polygon = self.get_polygon(polygon_type, landmark, resolution)
            polygon_list.append(polygon)
        
        mask_list = []
        masked_real_list = []
        real_bits = np.full([resolution, resolution],
                            CONTEXT_BIT,
                            dtype=np.uint8)

          
        # make obs mask list
        for target_domain in target_domain_set:
            mask_type = self.domain_lookup[target_domain, 2]

            polygon = polygon_list[mask_type]
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
            
            # draw polygon on the obs image
            mask = real_bits.copy()
            cv2.fillPoly(mask, polygon, MASK_BIT)
            mask_list.append(Image.fromarray(np.int8(mask)))

        sample['masked_real_list'] = masked_real_list
        sample['mask_list'] = mask_list
        sample['source_domain'] = source_domain
        sample['target_domain_list'] = target_domain_set
        return sample

    def get_target_domain(self, source_domain):
        """Make agumented domain one hot vector

        Args:
            source_domain (tensor) : [batch_size]
                                     source domain id
        """
        target_domain = self.augmented_domain[source_domain]
        target_domain = target_domain.flatten()
        target_domain = [x for x in target_domain if x != source_domain]
        target_domain.append(source_domain)
        return target_domain

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
        masked_real_list = []
        for masked_real in sample['masked_real_list']:
            masked_real = F.normalize(masked_real, self.mean, self.std)
            masked_real_list.append(masked_real)
        sample['masked_real_list'] = masked_real_list
                
        return sample

    def __repr__(self):
        return self.__class__.__name__ + \
            '(mean={0}, std={1})'.format(self.mean, self.std)
    
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for elem in ['image', 'attr', 'masked_real_list', 'mask_list']:
            if elem == 'attr':
                tmp = sample['attr']
                sample[elem] = torch.from_numpy(tmp).float().squeeze()

            elif elem == 'mask_list':
                mask_list = []
                for sub_mask_list in sample['mask_list']:
                    sub_list = []
                    for mask in sub_mask_list:
                        sub_list.append(F.to_tensor(mask))
                    mask_list.append(sub_list)
                sample[elem] = mask_list
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
        return self.__class__.__name__ + '()'
    
class ToTensor2(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for elem in ['image', 'attr', 'masked_real_list', 'mask_list']:
            if elem == 'attr':
                tmp = sample['attr']
                sample[elem] = torch.from_numpy(tmp).float().squeeze()

            elif elem == 'mask_list':
                tmp = sample['attr']
                mask_list = []
                for mask in sample['mask_list']:
                    mask_list.append(F.to_tensor(mask))
                sample[elem] = mask_list
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
        return self.__class__.__name__ + '()'