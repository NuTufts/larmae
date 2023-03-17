import os,sys
import torch

def save_checkpoint(state, is_best, p, tag=None):

    stem = "checkpoint"
    if tag is not None:
        stem += ".%s"%(tag)

    filename = "%s.%dth.tar"%(stem,p)
    torch.save(state, filename)
    if is_best:
        bestname = "model_best"
        if tag is not None:
            bestname += ".%s"%(tag)
        bestname += ".tar"
        shutil.copyfile(filename, bestname )

def load_model_checkpoint(checkpoint, model_key, remove_ddp_prefix=False):
    
    if not os.path.exists(str(checkpoint)):
        raise ValueError("Could not load the checkpoint file give: %s"%(checkpoint))
    loc = {}
    for i in range(20):
        loc["cuda:%d"%(i)] = "cpu"
    data = torch.load( checkpoint, map_location=loc )
    print("checkpoint file contents: ",data.keys())
    model_state = data[model_key]
    rename_dict = {}
    for k,t in model_state.items():
        if remove_ddp_prefix and "module." in k:
            k_new = k.replace("module.","")
            rename_dict[k_new] = t
        else:
            rename_dict[k] = t    
    return rename_dict
                    
