"""Test script."""

from torchvision import transforms
from torch.utils.data import DataLoader

from model.model import Generator, Discriminator
from model.stargan_model import StarGenerator, StarDiscriminator
from util.datasets import VGGFace2Dataset

from util.custom_transforms import Normalize, ToTensor


landmark_info_path = './dataset/VGGFACE2/train/all_loose_landmarks_256.csv'
identity_info_path = './dataset/VGGFACE2/identity_info.csv'
filtered_list = './dataset/VGGFACE2/train/all_filtered_results.csv'
transform = transforms.Compose([Normalize(),
                                ToTensor()])

batch_size = 2
num_classes = 2
num_attrs = 1
resolutions_to = [4, 8, 8, 16, 16, 32, 32]
levels = [1, 1.125, 2, 2.5, 3, 3.5, 4]
data_shape = [batch_size, 3, 32, 32]

G = Generator(data_shape, use_mask=False, use_attrs=True,
              num_attrs=num_attrs, latent_size=256)
D = Discriminator(data_shape, use_attrs=True,
                  num_attrs=num_attrs, latent_size=256)

for res, lev in zip(resolutions_to, levels):
    dataset = VGGFace2Dataset('./dataset/VGGFACE2/train', res,
                              landmark_info_path, identity_info_path,
                              filtered_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample = iter(dataloader).next()  # noqa: B305
    print(f"resolution: {res}, image: {sample['image'].shape}, \
            attr: {sample['attr'].shape}")

    # Generator
    if isinstance(lev, int):
        fake_image1 = G(sample['image'], sample['attr'], None, lev)
        assert list(fake_image1.shape) == [batch_size, 3, res, res]
    else:
        fake_image2 = G(sample['image'], sample['attr'], None, lev)
        assert list(fake_image2.shape) == [batch_size, 3, res, res]

    if isinstance(lev, int):
        cls1, attr1 = D(sample['image'], lev)  # training state
        assert list(cls1.shape) == [batch_size, 1]
        assert list(attr1.shape) == [batch_size, num_attrs]
    else:
        cls2, attr2 = D(sample['image'], lev)  # transition state
        assert list(cls2.shape) == [batch_size, 1]
        assert list(attr2.shape) == [batch_size, num_attrs]


dataset = VGGFace2Dataset('./dataset/VGGFACE2/train', 128,
                          landmark_info_path, identity_info_path,
                          filtered_list, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

sample = iter(dataloader).next()  # noqa: B305

S_G = StarGenerator(c_dim=1)
S_D = StarDiscriminator(c_dim=1)

img = sample['image']
c = sample['attr']

fake_img = S_G(img, c)
assert list(fake_img.shape) == [batch_size, 3, 128, 128]

_, out_cls = S_D(fake_img)
assert list(out_cls.shape) == [batch_size, 1]
print(out_cls.shape)
