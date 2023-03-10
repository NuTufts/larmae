import os,sys,time
import numpy as np
import torch
sys.path.append("/home/twongjirad/working/larbys/larmae/vit-pytorch/")
from vit_pytorch import ViT, MAE

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from larmae_dataset import larmaeDataset

print(ViT)

DEVICE = torch.device("cuda")

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
    
NITERS = 100000
LR = 1.0e-4
weight_decay=1.0e-6
batch_size = 8
test = larmaeDataset( 'tmp.yaml' )
print("NENTRIES: ",len(test))
shuffle = True

optimizer = torch.optim.AdamW(v.parameters(),
                              lr=LR,
                              weight_decay=weight_decay)

loader = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=shuffle)

start = time.time()
for iiter in range(NITERS):
    v.train()
    optimizer.zero_grad(set_to_none=True)
    print("====================================")
    print("ITER[%d]"%(iiter))
    
    batch = next(iter(loader))
    imgs = batch["img_plane2"].to(DEVICE)
    print("imgs: ",imgs.shape)
    
    maeloss = mae(imgs)
    
    print("Num nonzero patches: ",batch["num_nonzero_patches"])
    print("MAE-loss: ",maeloss)
    maeloss.backward()
    optimizer.step()
        
end = time.time()
elapsed = end-start
sec_per_iter = elapsed/float(NITERS)
print("sec per iter: ",sec_per_iter)
loader.dataset.print_status()
