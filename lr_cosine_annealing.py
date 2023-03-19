import os,sys
from math import cos
import numpy as np

def get_learning_rate_cosine_anealing_w_warmup( epoch, T, warmup_epochs,
                                                lr_min, lr_max, lr_warmup, lr_decay_factor ):
    if epoch<warmup_epochs:
        return lr_warmup

    Ti = T
    x_offset = warmup_epochs
    x = (epoch-x_offset)/Ti

    lr_factor = 1.0

    # if x, past the current period, warm restart.
    while x>1.0:
        x_offset += Ti # increment the offset
        if Ti<32:
            Ti *= 2 # we double the length of the next period until period is 16 epochs
        x = (epoch-x_offset) # remove the new offset
        x = x/Ti # normalize by new period
        lr_factor *= lr_decay_factor # we cut the range


    lr = lr_factor*(lr_min + 0.5*(lr_max-lr_min)*( 1+cos(x*np.pi) ))
    return lr
    

