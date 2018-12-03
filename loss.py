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
from torchvision.models import vgg16
from torch import autograd
from torch import nn

import util.util as util
from util.util import Gan
from util.util import GeneratorLoss
from util.util import DiscriminatorLoss
from util.util import Vgg16Layers


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
        self.alpha_recon = self.config.loss.alpha_recon

        self.lambda_GP = self.config.loss.lambda_GP
        self.lambda_recon = self.config.loss.lambda_recon
        self.lambda_feat = self.config.loss.lambda_feat
        self.lambda_bdy = self.config.loss.lambda_bdy
        self.lambda_attr = self.config.loss.lambda_attr
        self.lambda_cycle = self.config.loss.lambda_cycle

        self.g_losses = GeneratorLoss()
        self.d_losses = DiscriminatorLoss()

        self.gan = self.config.loss.gan
        self.create_loss_functions(self.gan)

        # for computing feature loss
        if self.config.loss.use_feat_loss:
            # Vgg16 ImageNet Pretrained Model
            self.vgg16 = Vgg16FeatureExtractor()

        self.register_on_gpu()

    def register_on_gpu(self):
        """Set vgg16 to cuda according to gpu availability."""
        if self.use_cuda:
            if self.config.loss.use_feat_loss:
                self.vgg16.cuda()

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
            cls_interpolates, attr_interpolates = D(interpolates, cur_level)
        else:
            # need cur_level
            cls_interpolates = D(interpolates, cur_level)

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
                                                       size_average=False) \
            / prediction.size(0)
        return attr_loss

    def calc_cycle_loss(self, G, cur_level, real, attr_real, mask, syn):
        """Calculate cycle consistency loss.

        Args:
            G: generator
            cur_level: progress indicator of progressive growing network
            real (tensor) : real images
            real_mask (tensor) : domain masks of real images
            obs_mask (tensor) : domain masks of observed images
        """
        N, C, H, W = real.shape

        if self.config.train.use_attr:
            pred_real = G(syn,
                          attr=attr_real,
                          mask=mask,
                          cur_level=cur_level)
        else:
            pred_real = G(syn,
                          attr=None,
                          mask=mask,
                          cur_level=cur_level)

        # L1 norm
        cycle_loss = torch.mean(torch.abs(real - pred_real))
        return cycle_loss

    def calc_feat_loss(self, real, syn):
        """Calculate feature loss.

        Args:
            real : real images
            syn : synthesized images

        """
        if self.config.loss.use_feat_loss is False:
            return 0

        # get activation of relu2_2
        N, C, H, W = real.shape
        # if H < 16 :
        #    return 0

        real_fmap = self.vgg16(real.detach(), Vgg16Layers.relu2_2)
        syn_fmap = self.vgg16(syn.detach(), Vgg16Layers.relu2_2)

        feat_loss = real_fmap - syn_fmap
        feat_loss = ((feat_loss.norm(2, dim=1) - 1.0) ** 2).mean()
        return feat_loss

    def calc_recon_loss(self, real, syn, mask):
        """Calculate reconstruction loss.

        Args:
            real : real images
            syn : synthesized images
            mask : binary mask

        """
        N, C, H, W = real.shape
        mask_ext = mask.repeat((1, C, 1, 1))

        # L1 norm
        alpha = self.alpha_recon
        recon_loss = (alpha * mask_ext * (real - syn)).norm(1) + \
                     ((1 - alpha) * (1 - mask_ext) * (real - syn)).norm(1)

        recon_loss = recon_loss/N

        return recon_loss

    def calc_bdy_loss(self, real, syn, mask):
        """Calculate boundary loss.

        Args:
            real : real images
            syn : synthesized images
            mask : binary mask

        """
        # blurring mask boundary
        N, C, H, W = mask.shape

        if H < 16:
            return 0

        mean_filter = MeanFilter(mask.shape, self.config.loss.mean_filter_size)
        if self.use_cuda:
            mean_filter.cuda()

        w1 = mean_filter(mask)
        w1 = w1 * (1 - mask)  # weights of mask range are 0
        w2 = mean_filter(1-mask)
        w2 = w2 * (mask)  # weights of non-mask range are 0
        w = w1 + w2
        w_ext = w.repeat((1, C, 1, 1))

        w_ext = util.tofloat(self.use_cuda, w_ext)

        bdy_loss = (w_ext * (real - syn)).norm(1)
        bdy_loss = bdy_loss.sum()/N

        return bdy_loss

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
        # cls_real = cls_real[:1, :]  # temporary code
        # cls_syn = cls_syn[:1, :] # temporary code

        # adversarial loss
        self.g_losses.g_adver_loss = self.calc_adver_loss(cls_syn, True, 1)
        # feature loss
        self.g_losses.feat_loss = self.calc_feat_loss(real, syn)
        # attribute loss
        self.g_losses.g_attr_loss = 0
        if self.config.train.use_attr:
            self.g_losses.g_attr_loss = self.calc_attr_loss(d_attr_obs,
                                                            attr_obs)

        self.g_losses.cycle_loss = 0
        self.g_losses.recon_loss = 0
        self.g_losses.bdy_loss = 0
        if use_mask:
            # reconstruction loss
            self.g_losses.recon_loss = self.calc_recon_loss(real, syn, mask)
            # boundary loss
            self.g_losses.bdy_loss = self.calc_bdy_loss(real, syn, mask)
        '''
        else:
            # cycle consistency loss
            self.g_losses.cycle_loss = self.calc_cycle_loss(G,
                                                            cur_level,
                                                            real,
                                                            attr_real,
                                                            mask,
                                                            syn)
        '''

        self.g_losses.g_loss = self.g_losses.g_adver_loss + \
            self.lambda_attr*self.g_losses.g_attr_loss + \
            self.lambda_recon*self.g_losses.recon_loss + \
            self.lambda_feat*self.g_losses.feat_loss + \
            self.lambda_bdy*self.g_losses.bdy_loss + \
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
        self.d_losses.d_attr_loss = 0
        if self.config.train.use_attr:
            self.d_losses.d_attr_loss = self.calc_attr_loss(d_attr_real,
                                                            attr_real)

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


class Vgg16FeatureExtractor(nn.Module):
    """Vgg16FeatureExtractor classes.

    Attributes:
        vgg16_input_size : vgg16 input image size (default = 254)
        features : feature map list of vgg16

    """

    def __init__(self):
        """Class initializer."""
        super(Vgg16FeatureExtractor, self).__init__()

        self.vgg16_input_size = 254
        end_layer = Vgg16Layers.relu4_3
        features = list(vgg16(pretrained=True).features)[:end_layer]

        self.features = nn.ModuleList(features)

    def forward(self, x, extracted_layer=Vgg16Layers.relu2_2):
        """Forward.

        Args:
            x: extracted feature map
            extracted_layer: vgg16 layer number

        """
        if x.shape[2] < self.vgg16_input_size:
            x = self.upsample_Tensor(x, self.vgg16_input_size)
        elif x.shape[2] > self.vgg16_input_size:
            x = self.downsample_Tensor(x, self.vgg16_input_size)

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == extracted_layer:
                return x
        return x

    def downsample_Tensor(self, x, out_size):
        """Downsample_Tensor.

        Args:
            x: input images
            out_size: down sampled size

        """
        in_size = x.shape[2]

        if in_size < out_size:
            return self.upsample_Tensor(x, out_size)

        kernel_size = in_size // out_size

        if kernel_size == 0:
            return x
        # padding = in_size - out_size*kernel_size
        x = nn.functional.avg_pool2d(x, kernel_size)    # no overlap
        # x = m(x)

        return x

    def upsample_Tensor(self, x, out_size):
        """Upsample Tensor.

        Args:
            x: input images
            out_size: up sampled size

        """
        in_size = x.shape[2]

        if in_size >= out_size:
            return self.downsample_Tensor(x, out_size)

        scale_factor = out_size // in_size

        if (out_size % in_size) != 0:
            scale_factor += 1

        x = nn.functional.upsample(x, scale_factor=scale_factor)

        return x


class MeanFilter(nn.Module):
    """MeanFilter classes.

    Attributes:
        filter : mean filter (convolution module)

    """

    def __init__(self, shape, filter_size):
        """Class initializer."""
        super(MeanFilter, self).__init__()

        self.filter = nn.Conv2d(shape[1],
                                shape[1],
                                filter_size,
                                stride=1,
                                padding=filter_size//2)

        init_weight = 1.0 / (filter_size*filter_size)
        nn.init.constant_(self.filter.weight, init_weight)

    def forward(self, x):
        """Forward.

        Args:
            x: input images

        """
        x = self.filter(x)
        return x
