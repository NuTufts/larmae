import os,time,copy,sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larcv import larcv
larcv.load_pyutil
from ctypes import c_int
import yaml

from einops.layers.torch import Rearrange

class larmaeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_file):
        """
        Parameters:
        
        """
        
        with open(cfg_file, "r") as stream:
            try:
                allcfg = yaml.safe_load(stream)
                print(allcfg)
            except yaml.YAMLError as exc:
                print(exc)
                
        self.cfg = allcfg.get('larmaeDataset')                
        filelist = self.cfg.get('filelist',None)
        self.cropsize = self.cfg.get("crop_size",512)
        self.adc_threshold = self.cfg.get("adc_threshold",10.0)
        self.min_crop_pixels = self.cfg.get("min_crop_pixels",1000)
        self.patch_dims = self.cfg.get("patch_dims",16)
        self.vector_index = self.cfg.get("vector_index",0)
        self.nonzero_patch_threshold = self.cfg.get("nonzero_patch_threshold",10) # our of 1024 patches for 512x512 image
        
        self.tree = rt.TChain("extbnb_images")
        
        if filelist is not None:
            for ifile in filelist:
                self.tree.Add( ifile )

        self.nentries = self.tree.GetEntries()
        self._nloaded = 0
        print("larmaeDataset created. TChain=",self.tree)


    def __getitem__(self, idx):
        #print("larmaeDataset[",self,"].get[",idx,"]")
        worker_info = torch.utils.data.get_worker_info()

        okentry = False
        num_tries = 0
        ioffset = 0
        data = {}
        while not okentry:
            okentry = True
            entry = idx+ioffset
            if entry>=self.nentries:
                entry = 0
                ioffset = 0
            self.tree.GetEntry(entry)
            img = self.tree.img_v.at(self.vector_index).tonumpy()

            cropok = False
            croptries = 0
            while croptries<10 and not cropok:
                bounds_xmax = img.shape[0]-self.cropsize-1
                bounds_ymax = img.shape[1]-self.cropsize-1
                x = np.random.randint(0,bounds_xmax)
                y = np.random.randint(0,bounds_ymax)

                crop = img[x:x+self.cropsize,y:y+self.cropsize]
                if (crop[crop>self.adc_threshold]).sum()>self.min_crop_pixels:
                    data["img"] = np.expand_dims( crop, axis=0 ) # change to (1,H,W)
                    data["entry"] = entry

                    tensor = torch.from_numpy( np.expand_dims(data["img"],axis=0) ) # expand to (1,1,H,W)

                    with torch.no_grad():
                        #print("tensor: ",tensor.shape)
                        patches = self.chunk( tensor ) # returns (1,num_patches,patchdim1*patchdim2)
                        #print("patches: ",patches)
                        patchsum = torch.sum(patches,2).squeeze() # (num_patches)
                        #print("patchsum: ",patchsum)
                        num_non_zero = (patchsum>10).sum().item() # float
                        #print(num_non_zero)
                        data["num_nonzero_patches"] = num_non_zero
                        if num_non_zero >= self.nonzero_patch_threshold:
                            cropok = True
                            okentry = True
                    # scale
                    data["img"] = np.clip( (data["img"]-50.0)/50.0, -1.0, 5.0 )
                    
            ioffset += 1
            num_tries += 1

        self._nloaded += 1

        return data


    def __len__(self):
        return self.nentries

    def chunk(self,x):
        #print(x.shape)
        layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_dims, p2 = self.patch_dims)
        x = layer(x)
        #print(x.shape)
        
        return x

    def print_status(self):
        print("larmaeDataset")
        print("  number loaded: ",self._nloaded)

    
if __name__ == "__main__":
    
    import time
    from larcv import larcv
    larcv.load_pyutil()

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
    
    niter = 10
    batch_size = 4
    test = larmaeDataset( 'tmp.yaml' )
    print("NENTRIES: ",len(test))
    shuffle = True
    
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=shuffle)
                                         #collate_fn=larmDataset.collate_fn)

    start = time.time()
    for iiter in range(niter):
        batch = next(iter(loader))
        print("====================================")
        #print(batch)
        print("ITER[%d]"%(iiter))
        print(" keys: ",batch.keys())
        for name,d in batch.items():
            if type(d) is torch.Tensor:
                print("  ",name,"-[array]: ",d.shape)
            elif type(d) is list:
                print("  ",name,"-list of n=",len(d))
            else:
                print("  ",name,"-[non-array]: ",type(d))

        # test the patch making routines
        #x = test.chunk( batch['img'] )
        #y = torch.sum(x,2)
        #for b in range(y.shape[0]):
        #    print(" batch[",b,"]: non-zero patches = ",(y[b,:]>10).sum())
        #print(y.shape)
        print(batch["num_nonzero_patches"])
        
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    loader.dataset.print_status()

    
