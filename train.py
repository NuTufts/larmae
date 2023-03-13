import os,sys,time
import numpy as np
import torch
sys.path.append("/home/twongjirad/working/larbys/larmae/vit-pytorch/")
from vit_pytorch import ViT, MAE

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from larmae_dataset import larmaeDataset

from calc_accuracies import calc_zero_vs_nonzero_accuracy,calc_occupied_pixel_accuracies
from utils import save_checkpoint

import wandb

START_ITER = 0
NITERS = 100000
LR = 1.0e-4
weight_decay=1.0e-6
batch_size = 8

logged_list = ['mse_zero','mse_nonzero','zero2zero','zero2occupied','occupied2zero','occupied2occupied']

wandb.init(project="larmae-dev",
           config={
               "learning_rate": LR,
               "batch_size":batch_size
           })

DEVICE = torch.device("cuda")
NITERS_PER_CHECKPOINT=10

v = ViT(
    image_size = 512,
    channels = 1,
    patch_size = 16,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
).to(DEVICE)

mae = MAE(
    encoder = v,
    masking_ratio = 0.10,
    decoder_dim = 512,
    decoder_depth = 6
).to(DEVICE)

mae_ntrainable  = sum(p.numel() for p in mae.parameters() if p.requires_grad)

if False:
    for n,p in mae.named_parameters():
        if p.requires_grad:
            print(n," ",p.shape)

print("MAE num trainable pars: ",mae_ntrainable)
      

cfg = """\
larmaeDataset:
  filelist:
    - test.root
  crop_size: 512
  adc_threshold: 10.0
  min_crop_pixels: 1000
"""

with open('tmp.yaml','w') as f:
    print(cfg,file=f)

test = larmaeDataset( 'tmp.yaml' )
print("NENTRIES: ",len(test))
shuffle = True

optimizer = torch.optim.AdamW(v.parameters(),
                              lr=LR,
                              weight_decay=weight_decay)

loader = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=shuffle)

start = time.time()
for iiter in range(START_ITER,START_ITER+NITERS):
    v.train()
    optimizer.zero_grad(set_to_none=True)
    print("====================================")
    print("ITER[%d]"%(iiter))
    
    batch = next(iter(loader))
    imgs = batch["img_plane2"].to(DEVICE)
    print("imgs: ",imgs.shape)
    
    maeloss, pred_masked, true_masked, masked_indices = mae(imgs,return_outputs=True)
    #print(masked_indices.shape)
    #print(masked_indices)
    #print(pred_masked.shape)
    #print(true_masked.shape)
    
    print("Num nonzero patches: ",batch["num_nonzero_patches"])
    print("MAE-loss: ",maeloss)
    maeloss.backward()
    optimizer.step()

    # accuracy calculations
    acc = {}
    with torch.no_grad():
        acc_dict1 = calc_zero_vs_nonzero_accuracy( pred_masked.detach(), true_masked.detach() )
        acc.update(acc_dict1)

        acc_dict2 = calc_occupied_pixel_accuracies( pred_masked.detach(), true_masked.detach() )
        acc.update(acc_dict2)


    logged = {}
    for x in logged_list:
        logged[x] = acc[x]
    logged["loss"] = maeloss.detach().cpu().item()
    
    wandb.log( logged )

    if iiter>START_ITER and iiter%NITERS_PER_CHECKPOINT==0:
        print("Save Checkpoint")
        save_checkpoint( {"iter":iiter,
                          "state_mae":mae.state_dict(),
                          "optimizer":optimizer.state_dict()},
                        False, iiter, tag="larmae" )
    
    
end = time.time()
elapsed = end-start
sec_per_iter = elapsed/float(NITERS)
print("sec per iter: ",sec_per_iter)
loader.dataset.print_status()

wandb.finish()
