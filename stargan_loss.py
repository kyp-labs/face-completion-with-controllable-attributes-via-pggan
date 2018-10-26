"""loss.py.

This module includes following loss related classes.

1. FaceGenLoss class :
   calculates 5 types losses of generator and discriminator.

   - adversarial loss : gan, wgan gp
   - reconstruction loss
   - boundary loss
   - feature loss
   - attribute loss

2. Vgg16FeatureExtractor class :
    has a VGG16 model pretrained ImageNet
    and a method of extraction of any feature map.

3. MeaFilter class :
    supports mean filtering of any image.

"""

import torch
import torch.nn.functional as F
from torch import autograd

import util.util as util
from util.util import Gan
from util.util import GeneratorLoss
from util.util import DiscriminatorLoss


class FaceGenLoss():
    """FaceGenLoss classes.

    Attributes:
        pytorch_loss_use : flag for use of PyTorch loss function
        config : configuration object
        use_cuda : flag for cuda use
        gpu : # of gpus
        alpha_adver_loss_syn : weight of syn images' loss of D
        alpha_recon : weight for mask area of reconstruction loss
        lambda_GP : weight of gradient panelty
        lambda_recon :weight of reconstruction loss
        lambda_feat : weight of feature loss
        lambda_bdy : weight of boundary loss
        lambda_attr : weight of attribute loss
        g_losses : losses of generator
        d_losses : losses of discriminator
        gan : type of gan {wgan gp, lsgan, gan}
        vgg16 : VGG16 feature extractor
        adver_loss_func : adversarial loss function
        attr_loss_func : attribute loss function

    """

    def __init__(self, config, use_cuda=False, gpu=-1):
        """Class initializer.

        Steps:
            1. Read loss params from self.config.py
            2. Create loss functions
            3. Create VGG16 model and feature extractor

        """
        self.config = config
        self.pytorch_loss_use = True

        self.use_cuda = use_cuda
        self.gpu = gpu

        self.alpha_adver_loss_syn = self.config.loss.alpha_adver_loss_syn
        self.lambda_GP = self.config.loss.lambda_GP
        self.lambda_attr = self.config.loss.lambda_attr
        self.lambda_cycle = self.config.loss.lambda_cycle

        self.g_losses = GeneratorLoss()
        self.d_losses = DiscriminatorLoss()

        self.gan = self.config.loss.gan
        self.create_loss_functions(self.gan)

    def create_loss_functions(self, gan):
        """Create loss functions.

        1. create adversarial loss function
        2. create attribute loss function

        Args:
            gan: type of gan {wgan gp, lsgan, gan}

        """
        # adversarial loss function
        if gan == Gan.sngan:
            self.adver_loss_func = lambda p, t: (-2.0*t+1.0) * torch.mean(p)
        elif gan == Gan.wgan_gp:
            self.adver_loss_func = lambda p, t: (-2.0*t+1.0) * torch.mean(p)
        elif gan == Gan.lsgan:
            self.adver_loss_func = lambda p, t: torch.mean((p-t)**2)
        elif gan == Gan.gan:  # 1e-8 torch.nn.BCELoss()
            if self.pytorch_loss_use:
                self.adver_loss_func = torch.nn.BCELoss()
            else:
                self.adver_loss_func = \
                    lambda p, t: -(torch.mean(t*torch.log(p+1e-8)) +
                                   torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % gan)

    def calc_adver_loss(self, prediction, target, w=1.0):
        """Calculate adversarial loss.

        Args:
            prediction: prediction of discriminator
            target: target label {True, False}
            w: weight of adversarial loss

        """
        if self.gan == Gan.gan and self.pytorch_loss_use:
            mini_batch = prediction.shape[0]
            if target is True:
                target = util.tofloat(self.use_cuda, torch.ones(mini_batch))
            else:
                target = util.tofloat(self.use_cuda, torch.zeros(mini_batch))
            return self.adver_loss_func(prediction, target)
        else:
            return self.adver_loss_func(prediction, target)

    def calc_gradient_penalty(self, D, cur_level, real, syn):
        """Calc gradient penalty of wgan gp.

        Args:
            D: discriminator
            cur_level: progress indicator of progressive growing network
            real: real images
            syn: synthesized images

        """
        N, C, H, W = real.shape

        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N,
                             real.nelement()//N).contiguous().view(N, C, H, W)
        alpha = util.tofloat(self.use_cuda, alpha)

        syn = syn.detach()
        interpolates = alpha * real + (1.0 - alpha) * syn

        interpolates = util.tofloat(self.use_cuda, interpolates)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        if self.config.train.use_attr:
            # need cur_level
            cls_interpolates, attr_interpolates = D(interpolates)
        else:
            # need cur_level
            cls_interpolates = D(interpolates)

        cls_interpolates = cls_interpolates[:1, :]  # temporary code
        grad_outputs = util.tofloat(self.use_cuda,
                                    torch.ones(cls_interpolates.size()))
        gradients = autograd.grad(outputs=cls_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    
    def calc_attr_loss(self, prediction, target):
        """Calculate attribute loss.
        Args:
            attr_real: attribute of real images
            d_attr_real : classes for attributes of real images
            attr_obs : attributes of observed images
            d_attr_obs : classes for attributes of observed images
        """
        # cross entropy loss with logit
        attr_loss = F.binary_cross_entropy_with_logits(prediction,
                                                       target,
                                                       size_average=False)\
                                                       / prediction.size(0)

        return attr_loss
    
    def calc_cycle_loss(self, G, cur_level, real, syn, attr_real):
        """Calculate cycle consistency loss.

        Args:
            G: generator
            cur_level: progress indicator of progressive growing network
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            obs_mask (tensor) : domain masks of observed images

        """
        N, C, H, W = real.shape

        pred_real = G(syn, attr_real)

        # L1 norm
        cycle_loss = F.l1_loss(pred_real, real, size_average=True)
        return cycle_loss

    def calc_G_loss(self,
                    G,
                    cur_level,
                    real,
                    obs,
                    attr_real,
                    attr_obs,
                    mask,
                    syn,
                    cls_real,
                    cls_syn,
                    d_attr_real,
                    d_attr_obs,
                    use_mask):
        """Calculate Generator loss.

        Args:
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
            use_mask : flag for mask use in the model

        """
        # adversarial loss
        self.g_losses.g_adver_loss = self.calc_adver_loss(cls_syn, True, 1)
        # attribute loss
        self.g_losses.g_attr_loss = self.calc_attr_loss(d_attr_obs, attr_obs)
        # cycle consistency loss
        self.g_losses.cycle_loss = self.calc_cycle_loss(G,
                                                        cur_level,
                                                        real,
                                                        syn,
                                                        attr_real)

        self.g_losses.g_loss = self.g_losses.g_adver_loss + \
            self.lambda_attr*self.g_losses.g_attr_loss + \
            self.lambda_cycle*self.g_losses.cycle_loss
        return self.g_losses

    def calc_D_loss(self,
                    D,
                    cur_level,
                    real,
                    obs,
                    attr_real,
                    attr_obs,
                    mask, syn,
                    cls_real,
                    cls_syn,
                    d_attr_real,
                    d_attr_obs):
        """Calculate Descriminator loss.

        Args:
            D: discriminator
            cur_level: progress indicator of progressive growing network
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
        # adversarial loss
        self.d_losses.d_adver_loss_real = \
            self.calc_adver_loss(cls_real, True, 1.0)
        self.d_losses.d_adver_loss_syn = \
            self.calc_adver_loss(cls_syn, False, 1.0)
        self.d_losses.d_adver_loss = self.d_losses.d_adver_loss_real + \
            self.alpha_adver_loss_syn * self.d_losses.d_adver_loss_syn

        # attribute loss
        self.d_losses.d_attr_loss = self.calc_attr_loss(d_attr_real, attr_real)

        if self.gan == Gan.wgan_gp:
            self.d_losses.gradient_penalty = \
                self.calc_gradient_penalty(D,
                                           cur_level,
                                           real,
                                           syn)
        else:
            self.d_losses.gradient_penalty = 0.0

        self.d_losses.d_loss = self.d_losses.d_adver_loss + \
            self.lambda_attr*self.d_losses.d_attr_loss + \
            self.lambda_GP*self.d_losses.gradient_penalty
        return self.d_losses