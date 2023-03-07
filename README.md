# LArMAE

Experiments in using Masked Autoencoders for pre-traininged a Vision Transformer on EXTBNB data.

We will use this as a baseline for several experiments:

* Fine-tuning for Matt's network
* Fine-tuning for LArMatch
* Collaboration with Bill Freeman on unsupervised semantic segmentation
* Fine-tuning for DEiT

Run3 G1 EXTBNB sample has 34K files with about 15 events each.
If we aim for a crop size of 512x512, we will have about 2*4 images from each event.

This leads us to roughly an effective image sample size of 4 million images for each plane, a bit more for the Y-plane.

We use larbys/larcv Version 1 for handling microboone data.

## Steps

1. Generate small sample size, 10
2. Get ViT encoder, maybe a standard one from something like mm
3. Define de-coder, just a small number of blocks
4. Practice training on small sample or even single image.
5. Big training
6. Fine tuning on MC labeled SSNet
7. Write paper, publish weights.

## Code

The repository [lucidarains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) has ViT and a MAE wrapper of some sort. Easy peasy!

