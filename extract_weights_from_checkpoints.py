import os,sys
import torch
from utils import load_model_checkpoint

def process_one_file( checkpoint_file, outfilename, model_key="state_mae", overwrite=False ):
    if not overwrite and os.path.exists(outfilename):
        print(outfilename," already exists.  skipping")
        return

    model_weights = load_model_checkpoint( checkpoint_file, model_key )
    torch.save({"state_mae":model_weights}, outfilename)

if __name__ == "__main__":

    overwrite = True
    weight_dir = "arxiv/"
    arxiv = os.listdir(weight_dir)
    for darxiv in arxiv:
        subfolder = weight_dir+"/"+darxiv
        subdir = os.listdir(subfolder)
        for s in subdir:
            if "checkpoint" in s and ".tar" in s:
                checkpoint_file = subfolder+"/"+s
                outname = checkpoint_file.replace("checkpoint","modelweights")
                print("read in ",checkpoint_file," and output ",outname)
                if not overwrite and os.path.exists(outname):
                    continue
                process_one_file(checkpoint_file,outname,overwrite=overwrite)
                #sys.exit(0)
    
