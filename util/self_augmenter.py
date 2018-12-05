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
                mask,
                mask_type,
                source_domain, 
                target_domain):
        """Make agumented domain data

        Args:
            real : real images
            obs: observed images
            mask : observed data mask
            attr_obs : attributes of observed images
            source_domain (tensor) : [batch_size, 1]
                                     source domain id
            target_domain (tensor) : [batch_size, 1]
                                     target domain id
        """
        # binary mask
        N, C, H, W = mask.shape
        mask = util.tofloat(self.use_cuda, mask)
        mask = mask.repeat((1, C, 1, 1))
        
        ## augmented domain
        augmented_domain, augmented_domain_onehot = \
            self.get_augmented_domain(source_domain, target_domain, mask_type)
        augmented_domain = util.tofloat(self.use_cuda, augmented_domain)          
        augmented_domain_onehot = util.tofloat(self.use_cuda,
                                               augmented_domain_onehot) 
        # if source domain is same with target domain, 
        #                               don't augment real images.
        aug_idx = np.arange(1, N+1) * (source_domain != target_domain) - 1

        # real image
        # obs, syn = self.normalize(obs, syn)
        real[aug_idx] = (syn * mask + obs * (1 - mask))[aug_idx]

        # obs attribute

        attr_obs[aug_idx] = augmented_domain_onehot[aug_idx]
        
        # real attribute
        attr_real[aug_idx] = augmented_domain_onehot[aug_idx]

        # target domain
        target_domain[aug_idx] = augmented_domain[aug_idx]

        return real, attr_real, attr_obs, target_domain
 
    def normalize(self, obs, syn):        
        N, C, H, W = obs.shape
        obs_syn = torch.cat((obs, syn), 1)
        for i in range(N):
            obs_syn_min = torch.min(obs_syn[i])
            obs_syn_max = torch.max(obs_syn[i])
            obs[i] -= obs_syn_min
            obs[i] /= obs_syn_max
            syn[i] -= obs_syn_min
            syn[i] /= obs_syn_max

        return obs, syn
    
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