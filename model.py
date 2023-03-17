import os,sys
import yaml
sys.path.append("/n/home01/twongjirad/larmae/vit-pytorch/")
import torch
from vit_pytorch import ViT, MAE


def load_model( cfg, strict=False ):

    if type(cfg) is str:
        with open(cfg,'r') as f:
            print("loading model through config file")
            _cfg = yaml.safe_load(f)
    else:
        _cfg = cfg
    
    model_cfg = _cfg.get("model")
    if model_cfg is None:
        raise ValueError("could not find 'model' parameter group in config")
    
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
        decoder_depth = 6,
        weight_by_pixel=model_cfg.get("weight_by_model",True),
        nonzero_pixel_threshold=model_cfg.get("nonzero_pixel_threshold",-0.4)
    )

    checkpoint = model_cfg.get("checkpoint_file",None)
    if checkpoint is not None:
        if not os.path.exists(str(checkpoint)):
            raise ValueError("Could not load the checkpoint file give: %s"%(checkpoint))
        loc = {}
        for i in range(20):
            loc["cuda:%d"%(i)] = "cpu"
        data = torch.load( checkpoint, map_location=loc )
        print("checkpoint file contents: ",data.keys())
        missing, extra = mae.load_state_dict( data["state_mae"], strict=strict )

    return mae

