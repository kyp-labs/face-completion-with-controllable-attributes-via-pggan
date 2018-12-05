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

    def __init__(self, config, use_coda, self_augmenter):
        """Class initializer."""
        self.config = config
        self.use_cuda = use_coda
        self.use_mask = self.config.train.use_mask
        self.cur_resol = 0
        
        # self augmenetation
        self.self_augmenter = self_augmenter
        self.augmented_domain = self.self_augmenter.augmented_domain
        self.num_augmented_domain = self.self_augmenter.num_augmented_domain
        self.domain_lookup = self.self_augmenter.domain_lookup
        
        self.augmented_train = self.config.augment.train
        self.augmented_train_iter = self.config.augment.iter
        self.mask_type_list = self.config.augment.mask_type_list
        
        self.attribute_size = self.config.dataset.attibute_size
        # sample file name
        self.sample_file_name = self.config.dataset.sample_path
        if self.config.snapshot.gen_sample_from_file == False:
            self.sample_file_name =  self.config.dataset.sample_path_temp

    def make_sample_file(self, batch_size):
        """Merge filtered list and save to csv."""
        ds = self.config.dataset
        filtered_list = pd.read_csv(ds.filtering_path)
        filtered_list = filtered_list[filtered_list['category'] == 'Good']
        seed = random.randint(0, filtered_list.shape[0])
        sample_list = filtered_list.sample(batch_size, random_state = seed)       
        sample_list.to_csv(self.sample_file_name, index=False)
  
    def load_sample_data(self, cur_resol, batch_size):
        """Load train set.

        Args:
            resol: progress indicator of progressive growing network
            batch_size: flag for detaching syn image from generator graph

        """
        if self.config.snapshot.gen_sample_from_file == False or \
            os.path.exists(self.sample_file_name) is False:
            self.make_sample_file(batch_size)

        self.cur_resol = cur_resol
        self.batch_size = batch_size
        augmented_domain = util.tensor2numpy(self.use_cuda,
                                             self.augmented_domain).astype(int)
        transform_options = transforms.Compose([dt2.PermuteDomainPolygonMask(
                                                    self.attribute_size,
                                                    augmented_domain,
                                                    self.domain_lookup,
                                                    self.mask_type_list),
                                                dt2.ToTensor2(),
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
                                          func=dataset_func,
                                          attribute_size=\
                                              self.num_augmented_domain)

        # train_dataset & data loader
        self.sample_set = DataLoader(datasets, batch_size)
        return True
    
    def preprocess(self):
        """Set input type to cuda or cpu according to gpu availability."""
        self.real = util.tofloat(self.use_cuda, self.real)
        self.attr_real = util.tofloat(self.use_cuda, self.attr_real)
        self.mask = util.tofloat(self.use_cuda, self.mask)
        self.obs = util.tofloat(self.use_cuda, self.obs)
        self.attr_obs = util.tofloat(self.use_cuda, self.attr_obs)
 
    def snapshot(self,
                 cur_resol,
                 G):
        """Snapshot.

        Args:
            cur_resol: image resolution of current layer
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
            
            self.mask_list = sample_batched['mask_list']
            self.masked_real_list = sample_batched['masked_real_list']
            
            self.obs = sample_batched['image']
            self.source_domain = sample_batched['source_domain']
            self.target_domain_list = sample_batched['target_domain_list']

            batch = []
            num_target_domain = len(self.target_domain_list)
            batch_index = np.arange(self.batch_size)

            N, C, H, W = self.obs.shape
            
            for target_idx in range(num_target_domain):
                target_domain = self.target_domain_list[target_idx]
                # mask_type = self.domain_lookup[target_domain, 2]
                self.mask = self.mask_list[target_idx]
                
                self.attr_obs = torch.zeros_like(self.attr_real)
                self.attr_obs[batch_index, target_domain.data.numpy()] = 1
                
                self.preprocess()
               
                self.syn = G(self.obs, self.attr_obs, self.mask)
                masked = self.masked_real_list[target_idx].cpu().data.numpy()
                batch.append(masked)
                batch.append(self.syn.cpu().data.numpy()) # real

            batch = np.concatenate(batch, axis=3)
           
            samples += [batch]

        samples = np.concatenate(np.array(samples), axis=2)
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])
        # normalization (syn image part)
        for target_idx in range(num_target_domain):
            real_idx = target_idx*2 + 1
            samples[:, real_idx, :] -= np.min(samples[:, real_idx, :])
            samples[:, real_idx, :] /= np.max(samples[:, real_idx, :])
            syn_idx = real_idx + 1
            samples[:, syn_idx, :] -= np.min(samples[:, syn_idx, :])
            samples[:, syn_idx, :] /= np.max(samples[:, syn_idx, :])

        return samples

    def snapshot_old(self,
                 cur_resol,
                 G):
        """Snapshot.

        Args:
            cur_resol: image resolution of current layer
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
            # self.obs_attr_list = self.permute_attr(self.attr_real)
            
            self.mask_list = sample_batched['mask_list']
            self.masked_real_list = sample_batched['masked_real_list']
            
            self.obs = sample_batched['image']
            self.source_domain = sample_batched['source_domain']
            self.target_domain_list = sample_batched['target_domain_list']

            batch = []
            # mask_num = len(self.masked_real_list)
            num_target_domain = len(self.target_domain_list)
            batch_index = np.arange(self.batch_size)
            for mask_type in range(1):
                masked = self.masked_real_list[mask_type].cpu().data.numpy()
                batch.append(masked)
    
                N, C, H, W = self.obs.shape
                
                for target_idx in range(num_target_domain):
                    target_domain = self.target_domain_list[target_idx]
                    self.mask = self.mask_list[mask_type][target_idx]
                    
                    self.attr_obs = torch.zeros_like(self.attr_real)
                    self.attr_obs[batch_index, target_domain.data.numpy()] = 1
                    
                    self.preprocess()
                   
                    if self.use_mask:
                        self.syn = G(self.obs, self.mask, self.attr_obs)
                    else:
                        self.syn = G(self.obs, self.attr_obs)
                    
                    batch.append(self.syn.cpu().data.numpy()) # real

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