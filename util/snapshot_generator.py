"""snapshot.py.

This module includes Snapshot class
which makes snapshots of intermediate images, checkpoints and tensorboard log

"""
import os
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import util.custom_transforms as dt
import util.snapshot_transforms as dt2
import util.util as util
import pandas as pd


class SnapshotGenerator(object):
    """Snapshot classes.

    Attributes:
        use_cuda : flag for cuda use
        sample_dir : smaple snapshot directory

    """

    def __init__(self, config, use_coda):
        """Class initializer."""
        self.config = config
        self.use_cuda = use_coda
        self.use_mask = self.config.train.use_mask
        self.cur_resol = 0

        self.sample_file_name = self.config.dataset.sample_path
        if self.config.snapshot.gen_sample_from_file is False:
            self.sample_file_name = self.config.dataset.sample_path_temp

    def make_sample_file(self, batch_size):
        """Merge filtered list and save to csv."""
        ds = self.config.dataset
        filtered_list = pd.read_csv(ds.filtering_path)
        filtered_list = filtered_list[filtered_list['category'] == 'Good']
        seed = random.randint(0, filtered_list.shape[0])
        sample_list = filtered_list.sample(batch_size, random_state=seed)
        sample_list.to_csv(self.sample_file_name, index=False)

    def load_sample_data(self, cur_resol, batch_size):
        """Load train set.

        Args:
            resol: progress indicator of progressive growing network
            batch_size: flag for detaching syn image from generator graph

        """
        if self.config.snapshot.gen_sample_from_file is False or \
                os.path.exists(self.sample_file_name) is False:
            self.make_sample_file(batch_size)

        self.cur_resol = cur_resol
        self.batch_size = batch_size

        transform_options = transforms.Compose([dt2.PermutePolygonMask(),
                                                dt2.ToTensor(),
                                                dt2.Normalize(
                                                    mean=(0.5, 0.5, 0.5),
                                                    std=(0.5, 0.5, 0.5)),
                                                dt.Normalize(
                                                    mean=(0.5, 0.5, 0.5),
                                                    std=(0.5, 0.5, 0.5))])

        dataset_func = self.config.dataset.func
        ds = self.config.dataset
        datasets = util.call_func_by_name(data_dir=ds.data_dir,
                                          resolution=cur_resol,
                                          landmark_info_path=ds.landmark_path,
                                          identity_info_path=ds.identity_path,
                                          filtered_list=self.sample_file_name,
                                          transform=transform_options,
                                          func=dataset_func)

        # train_dataset & data loader
        self.sample_set = DataLoader(datasets, batch_size)
        return True

    def snapshot(self,
                 cur_resol,
                 G):
        """Snapshot.

        Args:
            global_it : global # of iterations through training
            it: current # of iterations in the phases of the layer
            total_it: total # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            cur_level: progress indicator of progressive growing network
            batch_size: minibatch size
            G: generator

        """
        assert self.cur_resol == cur_resol

        samples = []
        for _, sample_batched in enumerate(self.sample_set):
            if sample_batched['image'].shape[0] < self.batch_size:
                break

            # get a next batch - temporary code
            self.real = sample_batched['image']
            self.attr_real = sample_batched['attr']
            self.obs_attr_list = self.permute_attr(self.attr_real)

            self.obs_mask_list = sample_batched['obs_mask_list']
            self.masked_real_list = sample_batched['masked_real_list']

            self.obs = sample_batched['image']

            attr_size = self.attr_real.shape[1]

            batch = []
            mask_num = len(self.masked_real_list)
            print(mask_num)
            for i in range(mask_num):
                batch.append(self.masked_real_list[i].cpu().data.numpy())

                N, C, H, W = self.obs.shape

                for attr in range(attr_size):
                    self.obs_mask = self.obs_mask_list[i][attr]
                    # mask = self.obs_mask.repeat((1, C, 1, 1))
                    self.attr_obs = self.obs_attr_list[attr]
                    self.preprocess()

                    if self.use_mask:
                        self.syn = G(self.obs, self.obs_mask, self.attr_obs)
                    else:
                        self.syn = G(self.obs, self.attr_obs)

                    batch.append(self.syn.cpu().data.numpy())  # real

            batch = np.concatenate(batch, axis=3)

            samples += [batch]

        samples = np.concatenate(np.array(samples), axis=2)
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])
        # normalization (syn image part)
        samples[:, :1, :] -= np.min(samples[:, :1, :])
        samples[:, :1, :] /= np.max(samples[:, :1, :])
        samples[:, 1:, :] -= np.min(samples[:, 1:, :])
        samples[:, 1:, :] /= np.max(samples[:, 1:, :])

        return samples

    def permute_attr(self, attr_real):
        """Generate attributes of observed images.

            - change randomly chosen one attribute of 50% of real images

        Args:
            attr_real: attributes of real images
        """
        N, attr_size = attr_real.shape
        attr_list = []
        for i in range(attr_size):
            attr_obs = torch.zeros_like(attr_real)
            attr_obs[:, i] = 1
            attr_list.append(attr_obs)
        return attr_list

    def preprocess(self):
        """Set input type to cuda or cpu according to gpu availability."""
        self.real = util.tofloat(self.use_cuda, self.real)
        self.attr_real = util.tofloat(self.use_cuda, self.attr_real)
        self.obs_mask = util.tofloat(self.use_cuda, self.obs_mask)
        self.obs = util.tofloat(self.use_cuda, self.obs)
        self.attr_obs = util.tofloat(self.use_cuda, self.attr_obs)
