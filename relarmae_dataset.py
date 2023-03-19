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

class relarmaeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_file, seed=0 ):
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
        filelist = self.cfg.get('files',None)
        txtfile  = self.cfg.get('filelist',None)
        
        # every value is a numpy array. will this avoid reference counting problem?
        self.cropsize = np.array( self.cfg.get("crop_size",512), dtype=np.int32 )
        self.adc_threshold = np.array( self.cfg.get("adc_threshold",10.0) )
        self.min_crop_pixels = np.array( self.cfg.get("min_crop_pixels",1000) )
        self.patch_dims = self.cfg.get("patch_dims",16)
        self.vector_index = np.array( self.cfg.get("vector_index",0), dtype=np.int32)
        self.nonzero_patch_threshold = np.array( self.cfg.get("nonzero_patch_threshold",10) ) # our of 1024 patches for 512x512 image
        self.get_batch = True
        self.batch_size = self.cfg.get("batch_size",4)
        self.rng = np.random.RandomState(seed)
        self.offset = np.zeros(1)

        # create ROOT TTree
        self.tree = rt.TChain("extbnb_images")

        if filelist is not None:
            for ifile in filelist:
                self.tree.Add( ifile )
                
        if txtfile is not None:
            if os.path.exists(txtfile):
                with open(txtfile) as f:
                    ll = f.readlines()
                    for l in ll:
                        if os.path.exists(l.strip()):
                            #print("add: ",l.strip())
                            self.tree.Add( l.strip() )
        
        self.nentries = np.array( self.tree.GetEntries(), dtype=np.long )
        print("[RE]larmaeDataset created. TChain=",self.tree," nentries=",self.nentries)

    def randomize_offset(self):
        self.offset[0] = self.rng.randint(0,self.nentries,size=1)

        
    def __getitem__(self, idx):
        if self.get_batch:
            img_batch   = torch.zeros( ( self.batch_size, 1, self.cropsize, self.cropsize ), dtype=torch.float )
            entry_batch = torch.zeros( self.batch_size, dtype=torch.long )
            num_nonzero_batch = torch.zeros( self.batch_size, dtype=torch.float )

            self.randomize_offset()
            
            for ib in range(self.batch_size):
                img,entry,nonzero = self.get_one_cropped_image(ib)
                img_batch[ib] = img
                entry_batch[ib] = entry
                num_nonzero_batch[ib] = nonzero

            return img_batch,entry_batch,num_nonzero_batch
        else:
            return self.get_one_cropped_image(idx)

    def get_one_cropped_image(self, idx):
        #worker_info = torch.utils.data.get_worker_info()        
        #print("relarmaeDataset[",self,"].get[",idx,"]. workerid=",worker_info.id)
        #
        # we expect this function to run inside a forked process function
        # see relarmae_mp_dataloader.py
                
        # we use ROOT to get us an image.
        # we then produce a random crop.
        #  this crop needs to have enough patches that are nonzero
        #  so we loop up to 10 tries to get such a crop.
        #  if we fail, we move onto the next entry in the ROOT tree

        # avoid referencing dictionaries and lists
        # -- memory leak issue due to reference counts when multi-processing?
        # https://github.com/pytorch/pytorch/issues/13246
                
        # what we return, as part of a tuple
        img = None
        entry = None
        num_non_zero = None

        okentry = False # indicate if we have an entry that returned a good crop
        num_tries = 0 # number of entries we've looped through to find a good crop to return
        ioffset = int(self.offset[0]) # entry number offset we'll reference for sequentially filling a batch

        # loop over entries to find a good crop
        while not okentry:
            okentry = False
            entry = idx+ioffset # entry we'll read
            if entry>=self.nentries:
                # reset the offset to work at the beginning of the file
                entry = 0
                self.offset[0] = -idx
                ioffset = -idx
                
            # read data from file
            self.tree.GetEntry(entry)

            # get numpy array containing image
            img = np.copy(self.tree.img_v.at( int(self.vector_index) ).tonumpy_nocopy())

            # now we need to find a crop
            cropok = False  # have we found a good crop?
            croptries = 0   # number of tries for the current entry
            while croptries<10 and not cropok:
                croptries += 1
                # define crop boundary
                bounds_xmax = img.shape[0]-self.cropsize-1
                bounds_ymax = img.shape[1]-self.cropsize-1
                x = self.rng.randint(0,bounds_xmax)
                y = self.rng.randint(0,bounds_ymax)
                # make the crop
                crop = img[x:x+self.cropsize,y:y+self.cropsize]

                # now we need to decide if crop is good.
                # we chunk the image into the patches the ViT will see.
                # we need a certain number of patches to have a minimum number of pixels
                # and a minimum number of patches with nonzero sums
                if (crop[crop>self.adc_threshold]).sum()>self.min_crop_pixels:
                    #data["img"] = np.expand_dims( crop, axis=0 ) # change to (1,H,W)
                    cropresize = np.expand_dims( crop, axis=0 ) # change to (1,H,W)

                    tensor = torch.from_numpy( np.expand_dims(cropresize,axis=0) ) # expand to (1,1,H,W)

                    # calc nonzero patches: determines if we return crop
                    with torch.no_grad():
                        #print("tensor: ",tensor.shape)
                        patches = self._chunk( tensor ) # returns (1,num_patches,patchdim1*patchdim2)
                        #print("patches: ",patches)
                        patchsum = torch.sum(patches,2).squeeze() # (num_patches)
                        #print("patchsum: ",patchsum)
                        num_non_zero = (patchsum>10).sum().item() # float
                        #print(num_non_zero)
                        #data["num_nonzero_patches"] = num_non_zero
                        if num_non_zero >= self.nonzero_patch_threshold:
                            cropok = True
                            okentry = True

                            # it's good, so make final scaled image
                            #data["img"] = np.clip( (data["img"]-20.0)/50.0, -1.0, 5.0 )
                            del img
                            img = torch.from_numpy( np.clip( (cropresize-20.0)/50.0, -1.0, 5.0 ) )

            if not okentry:
                # will need to try again
                ioffset += 1
                self.offset[0] += 1
            num_tries += 1

        return img,entry,num_non_zero


    def __len__(self):
        return self.nentries

    def _chunk(self,x):
        #print(x.shape)
        layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_dims, p2 = self.patch_dims)
        x = layer(x)
        #print(x.shape)        
        return x

    def chunk(x,patch_dims):
        layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=patch_dims, p2=patch_dims)
        x = layer(x)
        return x
    
    def collate_fn(batch):
        x = torch.utils.data.dataloader.default_collate(batch)
        # for user ease
        return {"img":x[0],"entry":x[1],"num_nonzero_patches":x[2]}

    def collate_fn_batch(batch):
        return {"img":batch[0][0],"entry":batch[0][1],"num_nonzero_patches":batch[0][2]}
        
    
if __name__ == "__main__":
    
    import time
    from larcv import larcv
    larcv.load_pyutil()

    from psutil import Process    

    cfg = """\
larmaeDataset:
  filelist:
#    - /cluster/tufts/wongjiradlabnu/twongj01/larmae/dataprep/data/mcc9_v29e_dl_run3_G1_extbnb_dlana/larmaedata_run3extbnb_0000.root
#    - /n/home01/twongjirad/mcc9_v29e_dl_run3_G1_extbnb_dlana/train/larmaedata_run3extbnb_0000.root
#    - /n/home01/twongjirad/mcc9_v29e_dl_run3_G1_extbnb_dlana/train/larmaedata_run3extbnb_*.root
    - larmaedata_run3extbnb_0000.root
  crop_size: 512
  adc_threshold: 10.0
  min_crop_pixels: 1000
  vector_index: 0
  batch_size: 64
  return_batch: True
"""

    with open('tmp.yaml','w') as f:
        print(cfg,file=f)
    
    niter = 20
    test = relarmaeDataset( 'tmp.yaml' )
    print("NENTRIES: ",len(test))
    shuffle = False
    FAKE_NET_RUNTIME = 0.250
    
    loader = torch.utils.data.DataLoader(test,batch_size=1,
                                         shuffle=shuffle,
                                         prefetch_factor=2,
                                         #num_workers=2,
                                         collate_fn=relarmaeDataset.collate_fn_batch)

    dt_load = 0.0    
    start = time.time()
    for iiter in range(niter):

        print("====================================")
        print("ITER[%d]"%(iiter))

        dt_iter = time.time()
        batch = next(iter(loader))
        dt_iter = time.time()-dt_iter
        dt_load += dt_iter

        if FAKE_NET_RUNTIME>0:
            print("pretend network: lag=",FAKE_NET_RUNTIME)
            time.sleep( FAKE_NET_RUNTIME )        
        
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
        print(batch["entry"])
        print("main process: %.03f"%(Process().memory_info().rss/1.0e9)," GB")
        
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)

    print("loading time per iteration: ",dt_load/float(niter))    
    print("WALL sec per iter: ",sec_per_iter)


    
