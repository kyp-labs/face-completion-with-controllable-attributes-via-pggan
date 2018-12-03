"""Custom dataset classes."""

import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VGGFace2Dataset(Dataset):
    """VGGFace2 Dataset according to the resolution."""

    def __init__(self, data_dir, resolution, landmark_info_path,
                 identity_info_path, filtered_list, use_low_res=False,
                 transform=None):
        """Constructor.

        Args:
            data_dir (str): Directory path containing dataset.
            resolution (int): Specific resolution value to load.
            landmark_info (str): Path of the file having landmark information.
            identity_info (str): Path of the file having identity information.
            filtered_list (str): Path of the file having filtered list
                                 information.
            use_low_res (bool): Use low resolution images or not.
            transform: Augmentation options, Default is None.
                       (e.g. torchvision.transforms.Compose([
                                transform.CenterCrop(10),
                                transform.ToTensor(),
                                ]))
        """
        filtered_list = pd.read_csv(filtered_list)
        dir_list = os.listdir(os.path.join(data_dir, str(resolution)))
        if use_low_res:
            self.file_list = filtered_list[filtered_list['category'] !=
                                      'Removed'][['filename']]  # noqa=E128
        else:
            self.file_list = filtered_list[filtered_list['category'] ==
                                      'Good'][['filename']]  # noqa=E128

        def _name_to_path(filename):
            return os.path.join(data_dir, str(resolution), filename)

        self.file_list['filepath'] = self.file_list['filename'].apply(
            _name_to_path)

        landmark_info = pd.read_csv(landmark_info_path)
        landmark_info = landmark_info[landmark_info['NAME_ID']
                                      .str.contains('|'.join(dir_list))]
        identity_info = pd.read_csv(identity_info_path)
        identity_info = identity_info[identity_info['Class_ID']
                                      .str.contains('|'.join(dir_list))]
        identity_info[' Gender'] = identity_info[' Gender'].apply(
            lambda x: [0, 1] if x == ' f' else [1, 0])

        self.landmark_info = landmark_info
        self.identity_info = identity_info

        self.cls_to_gender = identity_info.set_index('Class_ID')[' Gender']\
                                          .to_dict()
        self.transform = transform

    def __getitem__(self, idx):
        """Getter.

        Args:
            idx (int): index of image list.

        Return:
            sample (dict): {str: array} formatted data for training.
        """
        pattern = re.compile('n[0-9]{6}/[0-9]{4}_[0-9]{2}')
        image_path = self.file_list['filepath'].iloc[idx]
        # For Windows OS
        image_path = image_path.replace("\\", "/")
        name_id = re.search(pattern, image_path)[0]

        cls_id = name_id.split('/')[0]
        gender = np.array([self.cls_to_gender[cls_id]])

        landmark = self.landmark_info[self.landmark_info['NAME_ID'] ==
                                      name_id].iloc[:, 2:].values.flatten()
        image_pil = Image.open(image_path)

        sample = {'image': image_pil, 'landmark': landmark, 'attr': gender}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):  # noqa: D105
        return len(self.file_list)
