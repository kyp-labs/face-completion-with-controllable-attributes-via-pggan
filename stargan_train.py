"""train.py.

This module includes the FaceGen class
which has a model and train methods implementing the paper
'High-Resolution Image Synthesis and
    Semantic Manipulation with Conditional GANs'.


Example:
    Run this module without options (:
        $ python train.py

Note that all configrations for FaceGen are in self.config.py.

"""

import os
import torch
import torch.optim as optim

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from model.stargan_model import StarGenerator, StarDiscriminator
import util.util as util
from util.custom_transforms import *
from util.util import Phase
from util.util import Mode
from util.replay import ReplayMemory
from util.stargan_snapshot import Snapshot

from stargan_loss import FaceGenLoss


class FaceGenStarGAN():
    """FaceGen Classes.

    Attributes:
        D_repeats : How many times the discriminator is trained per G iteration
        total_size : Total # of real images in the training
        train_size : # of real images to show before doubling the resolution
        transition_size : # of real images to show when fading in new layers
        mode : running mode {inpainting , generation}
        use_mask : flag for mask use in the model
        use_attr : flag for attribute use in the model
        dataset_shape : input data shape
        use_cuda : flag for cuda use
        G : generator
        D : discriminator
        optim_G : optimizer for generator
        optim_D : optimizer for discriminator
        loss : losses of generator and discriminator
        replay_memory : replay memory
        global_it : global # of iterations through training
        global_cur_nimg : global # of current images through training
        snapshot : snapshot intermediate images, checkpoints, tensorboard logs
        real : real images
        obs: observed images
        attr_real : attributes of real images
        attr_obs : attributes of observed images
        mask : binary mask
        syn : synthesized images
        cls_real : classes for real images
        cls_syn : classes for synthesized images
        d_attr_real : classes for attributes of real images
        d_attr_obs : classes for attributes of observed images

    """

    def __init__(self, config):
        """Class initializer.

        1. Read configurations from self.config.py
        2. Check gpu availability
        3. Create a model and training related objects
        - Model (Generator, Discriminator)
        - Optimizer
        - Loss and loss histories
        - Replay memory
        - Snapshot

        """
        self.config = config
        self.D_repeats = self.config.train.D_repeats
        self.total_size = int(self.config.train.total_size *
                              self.config.train.dataset_unit)
        self.train_size = int(self.config.train.train_size *
                              self.config.train.dataset_unit)
        self.transition_size = int(self.config.train.transition_size *
                                   self.config.train.dataset_unit)
        assert (self.total_size == (self.train_size + self.transition_size)) \
            and self.train_size > 0 and self.transition_size > 0

        # GPU
        self.check_gpu()

        self.mode = self.config.train.mode
        self.use_mask = self.config.train.use_mask
        self.use_attr = self.config.train.use_attr

        # Generator & Discriminator Creation
        image_size = self.config.train.net.min_resolution
        self.G = StarGenerator(conv_dim=64,
                               c_dim=self.config.dataset.attibute_size,
                               repeat_num=6)
        self.D = StarDiscriminator(image_size=image_size,
                                   conv_dim=64,
                                   c_dim=self.config.dataset.attibute_size,
                                   repeat_num=6)

        self.register_on_gpu()
        self.create_optimizer()

        # Loss
        self.loss = FaceGenLoss(self.config,
                                self.use_cuda,
                                self.config.env.num_gpus)

        # Replay Memory
        self.replay_memory = ReplayMemory(self.config,
                                          self.use_cuda,
                                          self.config.replay.enabled)
        self.global_it = 1
        self.global_cur_nimg = 1

        # restore
        self.snapshot = Snapshot(self.config, self.use_cuda)
        self.snapshot.prepare_logging()
        self.snapshot.restore_model(self.G, self.D, self.optim_G, self.optim_D)

    def train(self):
        """Training for progressive growing model.

        1. Calculate min/max resolution for a model
        2. for each layer
            2-1. for each phases
                1) first layer : {training}
                2) remainder layers : {transition, traning}
                3) optional : {replaying}
                do train one step

        """
        min_resol = int(np.log2(self.config.train.net.min_resolution))
        max_resol = int(np.log2(self.config.train.net.max_resolution))
        assert 2**max_resol == self.config.train.net.max_resolution  \
            and 2**min_resol == self.config.train.net.min_resolution \
            and max_resol >= min_resol >= 2

        from_resol = min_resol
        if self.snapshot.is_restored:
            from_resol = int(np.log2(self.snapshot._resolution))
            self.global_it = self.snapshot._global_it

        assert from_resol <= max_resol

        # layer iteration
        for R in range(from_resol, max_resol+1):

            # Resolution & batch size
            cur_resol = 2 ** R
            if self.config.train.forced_stop \
               and cur_resol > self.config.train.forced_stop_resolution:
                break

            batch_size = self.config.sched.batch_dict[cur_resol]
            assert batch_size >= 1

            train_iter = self.train_size//batch_size
            transition_iter = self.transition_size//batch_size
            assert (train_iter != 0) and (transition_iter != 0)

            print("********** New Layer [%d x %d] : batch_size %d **********"
                  % (cur_resol, cur_resol, batch_size))

            # Phase
            if R == min_resol:
                phases = {Phase.training: [1, train_iter]}
                phase = Phase.training
                total_it = train_iter
            else:
                phases = {Phase.transition: [1, transition_iter],
                          Phase.training:
                              [train_iter+1, train_iter + transition_iter]}
                phase = Phase.transition
                total_it = train_iter + transition_iter

            if self.snapshot.is_restored:
                phase = self.snapshot._phase

            # Iteration
            from_it = phases[phase][0]
            to_it = phases[phase][1]

            if self.snapshot.is_restored:
                from_it = self.snapshot._it + 1
                self.snapshot.is_restored = False

            print("from_it %d, total_it %d" % (from_it, total_it))

            cur_nimg = from_it*batch_size
            cur_it = from_it

            # load traninig set
            self.training_set = self.load_train_set(cur_resol, batch_size)
            if self.config.replay.enabled:
                self.replay_memory.reset(cur_resol)

            # Learningn Rate
            lrate = self.config.optimizer.lrate
            self.G_lrate = lrate.G_dict.get(cur_resol,
                                            self.config.optimizer.lrate.G_base)
            self.D_lrate = lrate.D_dict.get(cur_resol,
                                            self.config.optimizer.lrate.D_base)

            # Training Set
            replay_mode = False

            while cur_it <= total_it:
                for _, sample_batched in enumerate(self.training_set):
                    if sample_batched['image'].shape[0] < batch_size:
                        break

                    if cur_it > total_it:
                        break

                    # trasnfer tansition to training
                    if cur_it == to_it and cur_it < total_it:
                        phase = Phase.training

                    # calculate current level (from 1)

                    if phase == Phase.transition:
                        # transition [pref level, current level]
                        cur_level = float(R - min_resol + float(cur_it/to_it))
                    else:
                        # training
                        cur_level = float(R - min_resol + 1)

                    # get a next batch - temporary code
                    self.real = sample_batched['image']
                    self.attr_real = sample_batched['attr']
                    self.mask = sample_batched['mask']
                    N, H, W = self.mask.shape
                    self.mask = self.mask.reshape((N, 1, H, W))

                    if self.mode == Mode.inpainting:
                        self.obs = sample_batched['masked_image']
                    else:
                        self.obs = sample_batched['image']

                    self.attr_obs = self.generate_attr_obs(self.attr_real)
                    # self.attr_obs = self.attr_real

                    cur_nimg = self.train_step(batch_size,
                                               cur_it,
                                               total_it,
                                               phase,
                                               cur_resol,
                                               cur_level,
                                               cur_nimg)
                    cur_it += 1
                    self.global_it += 1
                    self.global_cur_nimg += 1

            # Replay Mode
            if self.config.replay.enabled:
                replay_mode = True
                phase = Phase.replaying
                total_it = self.config.replay.replay_count

                for i_batch in range(self.config.replay.replay_count):
                    cur_it = i_batch+1
                    self.real, self.attr_real, self.mask,
                    self.obs, self.attr_obs, self.syn \
                        = self.replay_memory.get_batch(cur_resol, batch_size)
                    if self.real is None:
                        break
                    self.syn = util.tofloat(self.use_cuda, self.syn)
                    cur_nimg = self.train_step(batch_size,
                                               cur_it,
                                               total_it,
                                               phase,
                                               cur_resol,
                                               cur_level,
                                               cur_nimg,
                                               replay_mode)

    def train_step(self,
                   batch_size,
                   cur_it,
                   total_it,
                   phase,
                   cur_resol,
                   cur_level,
                   cur_nimg,
                   replay_mode=False):
        """Training one step.

        1. Train discrmininator for [D_repeats]
        2. Train generator
        3. Snapshot

        Args:
            batch_size: batch size
            cur_it: current # of iterations in the phases of the layer
            total_it: total # of iterations in the phases of the layer
            phase: training, transition, replaying
            cur_resol: image resolution of current layer
            cur_level: progress indicator of progressive growing network
            cur_nimg: current # of images in the phase
            replay_mode: Memory replay mode

        Returns:
            cur_nimg: updated # of images in the phase

        """
        self.preprocess()

        # Training discriminator
        self.update_lr(cur_it, total_it, replay_mode)
        self.optim_D.zero_grad()
        self.forward_D(cur_level, detach=True, replay_mode=replay_mode)
        self.backward_D(cur_level)

        if self.config.replay.enabled and replay_mode is False:
            self.replay_memory.append(cur_resol,
                                      self.real,
                                      self.attr_real,
                                      self.mask,
                                      self.obs,
                                      self.attr_obs,
                                      self.syn.detach())

        # Training generator
        if cur_it % self.D_repeats == 0:
            # Training generator
            self.optim_G.zero_grad()
            self.forward_G(cur_level)
            self.backward_G(cur_level)

        # model intermediate results
        self.snapshot.snapshot(self.global_it,
                               cur_it,
                               total_it,
                               phase,
                               cur_resol,
                               cur_level,
                               batch_size,
                               self.real,
                               self.syn,
                               self.G,
                               self.D,
                               self.optim_G,
                               self.optim_D,
                               self.loss.g_losses,
                               self.loss.d_losses)
        cur_nimg += batch_size

        return cur_nimg

    def forward_G(self, cur_level):
        """Forward generator.

        Args:
            cur_level: progress indicator of progressive growing network

        """
        self.cls_syn, self.d_attr_obs = self.D(self.syn)

    def forward_D(self, cur_level, detach=True, replay_mode=False):
        """Forward discriminator.

        Args:
            cur_level: progress indicator of progressive growing network
            detach: flag whether to detach graph from generator or not
            replay_mode: memory replay mode

        """
        self.syn = self.G(self.obs, self.attr_real)

        self.cls_real, self.d_attr_real = self.D(self.real)
        self.cls_syn, self.d_attr_obs = self.D(
                self.syn.detach() if detach else self.syn)

    def backward_G(self, cur_level):
        """Backward generator."""
        self.loss.calc_G_loss(self.G,
                              cur_level,
                              self.real,
                              self.obs,
                              self.attr_real,
                              self.attr_obs,
                              self.mask,
                              self.syn,
                              self.cls_real,
                              self.cls_syn,
                              self.d_attr_real,
                              self.d_attr_obs,
                              self.use_mask)
        self.loss.g_losses.g_loss.backward()
        self.optim_G.step()

    def backward_D(self, cur_level, retain_graph=True):
        """Backward discriminator.

        Args:
            cur_level: progress indicator of progressive growing network
            retain_graph: flag whether to retain graph of discriminator or not

        """
        self.loss.calc_D_loss(self.D,
                              cur_level,
                              self.real,
                              self.obs,
                              self.attr_real,
                              self.attr_obs,
                              self.mask,
                              self.syn,
                              self.cls_real,
                              self.cls_syn,
                              self.d_attr_real,
                              self.d_attr_obs)

        self.loss.d_losses.d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()

    def preprocess(self):
        """Set input type to cuda or cpu according to gpu availability."""
        self.real = util.tofloat(self.use_cuda, self.real)
        self.attr_real = util.tofloat(self.use_cuda, self.attr_real)
        self.mask = util.tofloat(self.use_cuda, self.mask)
        self.obs = util.tofloat(self.use_cuda, self.obs)
        self.attr_obs = util.tofloat(self.use_cuda, self.attr_obs)

    def check_gpu(self):
        """Check gpu availability."""
        self.use_cuda = torch.cuda.is_available() \
            and self.config.env.num_gpus > 0
        if self.use_cuda:
            gpus = str(list(range(self.config.env.num_gpus)))
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    def register_on_gpu(self):
        """Set model to cuda according to gpu availability."""
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def load_train_set(self, resol, batch_size):
        """Load train set.

        Args:
            resol: progress indicator of progressive growing network
            batch_size: flag for detaching syn image from generator graph

        """
        crop_size = 178
        image_size = 128
        transform_options = transforms.Compose([PolygonMask(),
                                                RandomHorizontalFlip(),
                                                CenterCrop(crop_size),
                                                Resize(image_size),
                                                ToTensor(),
                                                Normalize(mean=(0.5,0.5,0.5),
                                                    std=(0.5,0.5,0.5))])

        dataset_func = self.config.dataset.func
        ds = self.config.dataset
        datasets = util.call_func_by_name(data_dir=ds.data_dir,
                                          resolution=resol,
                                          landmark_info_path=ds.landmark_path,
                                          identity_info_path=ds.identity_path,
                                          filtered_list=ds.filtering_path,
                                          transform=transform_options,
                                          func=dataset_func)

        # train_dataset & data loader
        return DataLoader(datasets, batch_size, True)

    def generate_attr_obs(self, attr_real):
        """Generate attributes of observed images.

            - change randomly chosen one attribute of 50% of real images

        Args:
            attr_real: attributes of real images
        """
        # attribute is a n dimension vector with a value 0/1 for each elements
        assert len(attr_real.shape) == 2

        N, attr_size = attr_real.shape
        attr_obs = attr_real.clone()

        batch_index = np.arange(N)
        batch_index = batch_index[np.random.rand(N) > 0.5]
        attr_index = np.random.randint(attr_size, size=len(batch_index))

        attr_obs[batch_index, attr_index] \
            = 1 - attr_real[batch_index, attr_index]

        return attr_obs

    def create_optimizer(self):
        """Create optimizers of generator and discriminator."""
        self.optim_G = optim.Adam(self.G.parameters(),
                                  lr=self.config.optimizer.lrate.G_base,
                                  betas=(self.config.optimizer.G_opt.beta1,
                                         self.config.optimizer.G_opt.beta2))
        self.optim_D = optim.Adam(self.D.parameters(),
                                  lr=self.config.optimizer.lrate.D_base,
                                  betas=(self.config.optimizer.D_opt.beta1,
                                  self.config.optimizer.D_opt.beta2))

    def rampup(self, cur_it, rampup_it):
        """Ramp up learning rate.

        Args:
            cur_it: current # of iterations in the phase
            rampup_it: # of iterations for ramp up

        """
        if cur_it < rampup_it:
            p = max(0.0, float(cur_it)) / float(rampup_it)
            p = 1.0 - p
            return np.exp(-p*p*5.0)
        else:
            return 1.0

    def rampdown_linear(self, cur_it, total_it, rampdown_it):
        """Ramp down learning rate.

        Args:
            cur_it: current # of iterations in the phasek
            total_it: total # of iterations in the phase
            rampdown_it: # of iterations for ramp down

        """
        if cur_it >= total_it - rampdown_it:
            return float(total_it - cur_it) / rampdown_it

        return 1.0

    def update_lr_old(self, cur_it, total_it, replay_mode=False):
        """Update learning rate.

        Args:
            cur_it: current # of iterations in the phasek
            total_it: total # of iterations in the phase
            replay_mode: memory replay mode

        """
        if replay_mode:
            return

        rampup_it = total_it * self.config.optimizer.lrate.rampup_rate
        rampdown_it = total_it * self.config.optimizer.lrate.rampdown_rate

        # learning rate rampup & down
        for param_group in self.optim_G.param_groups:
            lrate_coef = self.rampup(cur_it, rampup_it)
            lrate_coef *= self.rampdown_linear(cur_it,
                                               total_it,
                                               rampdown_it)
            param_group['lr'] = lrate_coef * self.G_lrate
            # print("learning rate %f" % (param_group['lr']))

        for param_group in self.optim_D.param_groups:
            lrate_coef = self.rampup(cur_it, rampup_it)
            lrate_coef *= self.rampdown_linear(cur_it,
                                               self.total_size,
                                               rampdown_it)
            param_group['lr'] = lrate_coef * self.D_lrate

    def update_lr(self, cur_it, total_it, replay_mode=False):
        """Update learning rate.

        Args:
            cur_it: current # of iterations in the phasek
            total_it: total # of iterations in the phase
            replay_mode: memory replay mode

        """
        if replay_mode:
            return

        # Decay learning rates.
        num_iters_decay = total_it//2
        lr_update_step = 1000
        if cur_it % lr_update_step == 0 \
            and cur_it > (total_it - num_iters_decay):
            self.G_lrate -= (self.G_lrate / float(num_iters_decay))
            self.D_lrate -= (self.D_lrate / float(num_iters_decay))

            for param_group in self.optim_G.param_groups:
                param_group['lr'] = self.G_lrate
            for param_group in self.optim_D.param_groups:
                param_group['lr'] = self.D_lrate

            print('Learning Rate, G: {}, D: {}.'.format(self.G_lrate, self.D_lrate))
