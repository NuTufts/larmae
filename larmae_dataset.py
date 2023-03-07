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

class larmaeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg_file):
        """
        Parameters:
        
        """
        
        with open(cfg_file, "r") as stream:
            try:
                allcfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        self.cfg = allcfg.get('larmaeDataset')                
        filelist = self.cfg.get('filelist',None)

        self.tree = rt.TChain("extbnb_images")
        
        if filelist is not None:
            for ifile in filelist:
                self.tree.Add( ifile )

        self.nentries = self.tree.GetEntries()
        self._nloaded = 0


    def __getitem__(self, idx):
        print("larmaeDataset.get[",idx,"]")
        worker_info = torch.utils.data.get_worker_info()

        okentry = False
        num_tries = 0
        ioffset = 0
        data = {"meta":[]}
        while not okentry:
            okentry = True
            entry = idx+ioffset
            if entry>=self.nentries:
                entry = 0
                ioffset = 0
            self.tree.GetEntry(entry)
            data["img_plane2"] = self.tree.img_v.at(2).tonumpy()
            #data["meta"].append( self.tree.meta_v.at(2) )
            ioffset += 1
            num_tries += 1

        self._nloaded += 1

        return data


    def __len__(self):
        return self.nentries

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
  cropsize: 512
    """

    with open('tmp.yaml','w') as f:
        print(cfg,file=f)
    
    niter = 10
    batch_size = 1
    test = larmaeDataset( 'tmp.yaml' )
    print("NENTRIES: ",len(test))
    
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size)
                                         #collate_fn=larmDataset.collate_fn)

    start = time.time()
    for iiter in range(niter):
        batch = next(iter(loader))
        print("====================================")
        print(batch)
        for ib in range(batch_size):
            print("ITER[%d]:BATCH[%d]"%(iiter,ib))
            print(" keys: ",batch.keys())
            for name,d in batch.items():
                if type(d) is torch.Tensor:
                    print("  ",name,"-[array]: ",d.shape)
                else:
                    print("  ",name,"-[non-array]: ",type(d))
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    loader.dataset.print_status()

    
