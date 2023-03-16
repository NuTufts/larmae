import os,sys
from math import cos
import numpy as np

def get_learning_rate_cosine_anealing_w_warmup( epoch, T, warmup_epochs,
                                                lr_min, lr_max, lr_warmup ):
    if epoch<warmup_epochs:
        return lr_warmup

    Ti = T
    x_offset = warmup_epochs
    x = (epoch-x_offset)/Ti

    lr_factor = 1.0

    # if x, past the current period, warm restart.
    while x>1.0:
        x = (epoch-x_offset)
        Ti *= 2 # we double the next period
        x = x/Ti
        lr_factor *= 0.707 # we cut the range by 1/sqrt(2)
        x_offset += Ti

    lr = lr_factor*(lr_min + 0.5*(lr_max-lr_min)*( 1+cos(x*np.pi) ))
    return lr
    

