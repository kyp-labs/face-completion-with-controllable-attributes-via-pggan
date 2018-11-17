from model.stargan_model import *


def test_pix_cls_stargan():
    # c_dim: the number of domains
    D = StarDiscriminator(c_dim=2)

    # sample image
    img = torch.ones([1, 3, 128, 128])

    # get results
    out_src, out_cls, out_pix_cls = D(img)
    assert list(out_pix_cls.shape) == [1, 2, 128, 128]


if __name__ == "__main__":
    test_pix_cls_stargan()
