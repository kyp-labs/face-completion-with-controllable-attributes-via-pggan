"""Training test code."""
import config
import numpy as np
# from train import FaceGen
from stargan_train import FaceGenStarGAN
import datetime as dt

if __name__ == "__main__":
    begin_time = dt.datetime.now()

    env = 'stargan'
    print('With stargan config,')
    cfg = config.StarGANConfig()

    print('Running FaceGen()...')
    np.random.seed(cfg.common.random_seed)
    facegen = FaceGenStarGAN(cfg)
    facegen.train()

    end_time = dt.datetime.now()

    print()
    print("FaceGen()", end_time)
    print("Running Time", end_time - begin_time)
    print('Exiting...')
