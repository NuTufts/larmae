import os,sys,time,argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import yaml

from torchvision import transforms, utils

#sys.path.append("/home/twongjirad/working/larbys/larmae/vit-pytorch/")
#sys.path.append("/cluster/tufts/wongjiradlabnu/twongj01/larmae/vit-pytorch/")
sys.path.append("/n/home01/twongjirad/larmae/vit-pytorch/")
from vit_pytorch import ViT, MAE

from larmae_dataset import larmaeDataset

from calc_accuracies import calc_zero_vs_nonzero_accuracy,calc_occupied_pixel_accuracies
from utils import save_checkpoint

import wandb

START_ITER = 0
NITERS = 100000
NITERS_PER_CHECKPOINT=10000
WANDB_PROJECT="larmae-dev"
LOG_WANDB = False
LR = 1.0e-4
weight_decay=1.0e-6
batch_size = 64
NITERS_PER_LOG = 10

logged_list = ['mse_zero','mse_nonzero','zero2zero','zero2occupied','occupied2zero','occupied2occupied']

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, backend, port):
    """
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def run(gpu,args):

    #========================================================
    # CREATE PROCESS
    rank = gpu
    print("START run() PROCESS: rank=%d gpu=%d"%(rank,gpu))
    setup( rank, args.gpus, "nccl", "12355" )
    #========================================================
    torch.manual_seed(rank)

    # Setup Wandb if the rank-0 process
    if rank==0 and LOG_WANDB:
        print("RAN-0 THREAD LOAD WANDB")
        wandb.init(project=WANDB_PROJECT,
                   config={
                       "learning_rate": LR,
                       "batch_size":batch_size} )
        sys.stdout.flush()

    # turn the config file into a yaml object
    #with open(args.config_file,'r') as f:
    #    cfg = yaml.safe_load(f)

    # set device
    torch.cuda.set_device(gpu)
    DEVICE = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")

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
        masking_ratio = 0.10,
        decoder_dim = 512,
        decoder_depth = 6
    ).to(DEVICE)

    mae_ntrainable  = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    print("MAE num trainable pars: ",mae_ntrainable)

    # Option to dump out parameter names and shape [for debug]
    if False:
        for n,p in mae.named_parameters():
            if p.requires_grad:
                print(n," ",p.shape)

    mae = nn.parallel.DistributedDataParallel(mae, device_ids=[gpu],find_unused_parameters=True)
    print("RANK-%d Loaded Model"%(rank))
    sys.stdout.flush()
    torch.distributed.barrier()

    dataset = larmaeDataset( args.config_file )
    print("RANK-%d Start Data Loader. Number of entries: ",len(dataset))
    shuffle = True

    optimizer = torch.optim.AdamW(v.parameters(),
                                  lr=LR,
                                  weight_decay=weight_decay)

    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

    start = time.time()
    for iiter in range(START_ITER,START_ITER+NITERS):
        v.train()
        optimizer.zero_grad(set_to_none=True)
        print("====================================")
        print("ITER[%d]"%(iiter))
    
        batch = next(iter(loader))
        imgs = batch["img"].to(DEVICE)
        print("imgs: ",imgs.shape)
    
        maeloss, pred_masked, true_masked, masked_indices = mae(imgs,return_outputs=True)
        #print(masked_indices.shape)
        #print(masked_indices)
        #print(pred_masked.shape)
        #print(true_masked.shape)
    
        print("Num nonzero patches: ",batch["num_nonzero_patches"])
        print("MAE-loss: ",maeloss)
        maeloss.backward()
        optimizer.step()
        
        if iiter%NITERS_PER_LOG==0:
            # Log information about this iteration
            logged = {}
            
            # accuracy calculations
            acc = {}
            with torch.no_grad():
                acc_dict1 = calc_zero_vs_nonzero_accuracy( pred_masked.detach(), true_masked.detach() )
                acc.update(acc_dict1)
            
                acc_dict2 = calc_occupied_pixel_accuracies( pred_masked.detach(), true_masked.detach() )
                acc.update(acc_dict2)

            # grab items into 
            for x in logged_list:
                logged[x] = acc[x]

            # add the MSE loss
            logged["loss"] = maeloss.detach().cpu().item()

            # pass to wandb server
            if LOG_WANDB and rank==0:
                wandb.log( logged )

        # check pointing
        if rank==0 and iiter>START_ITER and iiter%NITERS_PER_CHECKPOINT==0:
            print("RANK-%d: Save Checkpoint"%(rank))
            save_checkpoint( {"iter":iiter,
                              "state_mae":mae.state_dict(),
                              "optimizer":optimizer.state_dict()},
                             False, iiter, tag="larmae" )
    
    
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(NITERS)
    print("sec per iter: ",sec_per_iter)
    loader.dataset.print_status()

    if rank==0 and LOG_WANDB:
        wandb.finish()

    cleanup()
    
    return True


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, default="config.yaml", type=str,
                        help='configuration file [default: config.yaml]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--no-parallel',default=False,action='store_true',help='if provided, will run without distributed training')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    if args.no_parallel:
        print("RUNNING WITHOUT USING TORCH DDP")
        run( 0, args )
    else:
        mp.spawn(run, nprocs=args.gpus, args=(args,), join=True)

    print("DISTRIBUTED MAIN DONE")
    
if __name__ == '__main__':
    main()
    
