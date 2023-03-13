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
