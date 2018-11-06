from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from util.datasets import *
from util.custom_transforms import *
from model.stargan_model import *


crop_size = 128
image_size = 128

transform = transforms.Compose([PolygonMask(), RandomHorizontalFlip(),
                                CenterCrop(crop_size), Resize(image_size),
                                ToTensor(), Normalize(mean=(0.5,0.5,0.5),
                                    std=(0.5,0.5,0.5))])

landmark_info_path = './dataset/VGGFACE2/train/all_loose_landmarks_256.csv'
identity_info_path = './dataset/VGGFACE2/identity_info.csv'
filtered_list = './dataset/VGGFACE2/train/all_filtered_results.csv'

dataset = VGGFace2Dataset('./dataset/VGGFACE2/train', 128, landmark_info_path, identity_info_path, filtered_list, transform=transform)

dataloader = DataLoader(dataset, 2, shuffle=False)
sample = iter(dataloader).next()

img = sample['image'].float()
attr = sample['attr'].float()
mask = sample['obs_mask'].float()


# Only mask used
G = StarGenerator(c_dim=0, use_mask=True)
D = StarDiscriminator(c_dim=2)

fake_img = G(img, mask=mask)
assert list(fake_img.shape) == [2, 3, 128, 128]
out_src, out_cls = D(fake_img)
assert list(out_src.shape) == [2, 1, 2, 2]
assert list(out_cls.shape) == [2, 2]


# Attr + mask used
G = StarGenerator(c_dim=2, use_mask=True)
D = StarDiscriminator(c_dim=2)

fake_img = G(img, mask=mask, c=attr)
assert list(fake_img.shape) == [2, 3, 128, 128]
out_src, out_cls = D(fake_img)
assert list(out_src.shape) == [2, 1, 2, 2]
assert list(out_cls.shape) == [2, 2]
