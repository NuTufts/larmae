import os,sys
import torch
sys.path.append("/n/home01/twongjirad/larmae/vit-pytorch/")
from vit_pytorch import ViT, MAE

ckpt_file = sys.argv[1]

map_loc = {}
for i in range(10):
    map_loc["cuda:%d"%(i)] = "cpu"
    
data = torch.load( ckpt_file, map_location=map_loc )

print(data.keys())
modelstate = data["state_mae"]
print(modelstate.keys())

search_for = ["decoder.layers.1.1.norm.weight","decoder.layers.1.1.norm.bias"]

for k,v in modelstate.items():
    for s in search_for:
        if s in k:
            print(k)
            print(v)

if False:
    sys.exit(0)

v = ViT(
    image_size = 512,
    channels = 1,
    patch_size = 16,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

mae = MAE(
    encoder = v,
    masking_ratio = 0.50,
    decoder_dim = 512,
    decoder_depth = 6
)

print(mae)
