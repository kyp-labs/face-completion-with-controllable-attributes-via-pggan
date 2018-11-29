"""snapshot.py.

This module includes Snapshot class
which makes snapshots of intermediate images, checkpoints and tensorboard log

"""
import util.util as util
import torch
import numpy as np

class SelfAugmenter(object):
    """Snapshot classes.

    Attributes:
        use_cuda : flag for cuda use
        sample_dir : smaple snapshot directory

    """

    def __init__(self, config, use_coda):
        """Class initializer."""
        self.config = config
        self.use_cuda = use_coda
        
        self.augmented_train = self.config.augment.train
        self.augmented_train_iter = self.config.augment.iter
        self.mask_type_list = self.config.augment.mask_type_list
        
        self.num_mask = len(self.mask_type_list)
        self.make_augmented_domain()
        
    def make_augmented_domain(self):

        num_domain = self.config.dataset.attibute_size
        
        self.augmented_domain = torch.zeros((num_domain,
                                             num_domain,
                                             self.num_mask))

        self.num_augmented_domain = num_domain + \
            num_domain*(num_domain-1)*(self.num_mask-1)
            
        self.augmented_domain_onehot = torch.zeros((num_domain,
                                                    num_domain,
                                                    self.num_mask,
                                                    self.num_augmented_domain))
            
        self.domain_lookup = np.zeros((self.num_augmented_domain, 3),
                                      dtype = np.uint8)
        
        new_id = num_domain
        for source in range(num_domain):
            for target in range(num_domain):
                for mask_type in range(self.num_mask):
                    if source == target:
                        domain_id = target
                    elif mask_type == 0: # face mask
                        domain_id = target
                    else:
                        domain_id = new_id
                        new_id += 1
 
                    self.augmented_domain[source, target, mask_type] = \
                        domain_id
                    self.augmented_domain_onehot[source,
                                                 target,
                                                 mask_type,
                                                 domain_id] = 1

                    mask_info = 0 if source == target else mask_type                        
                    self.domain_lookup[domain_id,0] = source
                    self.domain_lookup[domain_id,1] = target
                    self.domain_lookup[domain_id,2] = mask_info
                                                 
        
    def augment(self,
                real,
                obs,
                syn,
                attr_real,
                attr_obs,
                real_mask,
                obs_mask,
                mask_type,
                source_domain, 
                target_domain):
        """Make agumented domain data

        Args:
            real : real images
            obs: observed images
            obs_mask : observed data mask
            attr_obs : attributes of observed images
            source_domain (tensor) : [batch_size, 1]
                                     source domain id
            target_domain (tensor) : [batch_size, 1]
                                     target domain id
        """
        # binary mask
        N, C, H, W = obs_mask.shape
        target_bits = \
            target_domain.reshape(N, 1, 1, 1).repeat((1, C, H, W))
        obs_mask = util.tofloat(self.use_cuda, obs_mask)
        target_bits = util.tofloat(self.use_cuda, target_bits)
        target_mask = util.tofloat(self.use_cuda, obs_mask == target_bits)
        
        ## augmented domain
        augmented_domain, augmented_domain_onehot = \
            self.get_augmented_domain(source_domain, target_domain, mask_type)
            
        augmented_target_bits = \
            augmented_domain.reshape(N, 1, 1, 1).repeat((1, C, H, W))
        
        # obs mask
        context_mask = 1 - target_mask
        obs_mask = obs_mask * context_mask + \
            augmented_target_bits * target_mask
        
        # obs attribute
        attr_obs = augmented_domain_onehot

        # real image
        N, C, H, W = syn.shape
        target_mask = target_mask.repeat((1, C, 1, 1))
        context_mask = 1 - target_mask
        
        # if source domain and target domain is same, no augmentation
        real = obs * context_mask + syn * target_mask
        
        batch_index = np.arange(1, N+1) * (source_domain == target_domain) - 1
        real[batch_index] = obs[batch_index]
        
        # real mask
        real_mask = augmented_target_bits
        
        # real attribute
        attr_real = augmented_domain_onehot
        
        # target domain
        target_domain = augmented_domain
 
    def get_augmented_domain(self,
                             source_domain,
                             target_domain,
                             mask_type):
        """Make agumented domain one hot vector

        Args:
            source_domain (tensor) : [batch_size, 1]
                                     source domain id
            target_domain (tensor) : [batch_size, 1]
                                     target domain id
        """
        N = len(target_domain)
        new_domain = target_domain.clone()
        new_domain_onehot = torch.zeros((N, self.num_augmented_domain))
        
        for i in range(N):
            source = int(source_domain[i])
            target = int(target_domain[i])

            new_domain[i] = self.augmented_domain[source, target, mask_type]
            new_domain_onehot[i] = \
                self.augmented_domain_onehot[source, target, mask_type]
                
        return new_domain, new_domain_onehot