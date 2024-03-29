import os,sys,time,argparse,gc
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

from relarmae_dataset import relarmaeDataset
from relarmae_mp_dataloader import relarmaeMultiProcessDataloader

from lr_cosine_annealing import get_learning_rate_cosine_anealing_w_warmup
from calc_accuracies import calc_zero_vs_nonzero_accuracy,calc_occupied_pixel_accuracies
from utils import save_checkpoint
from model import load_model

import wandb

START_ITER = 0
NITERS = 10000
NITERS_PER_CHECKPOINT=2000
NITERS_PER_MODEL_LOG = 2000
NITERS_PER_LOG = 10
NITERS_PER_DATALOADER_RESET = 1000
WANDB_PROJECT="larmae-dev"
LOG_WANDB = False
LR = 1.0e-6
weight_decay=5.0e-2
batch_size = 64
num_workers=2
Tbase = 1.0
warmup_epochs = 2.0
lr_min = 1.0e-6
lr_max = 1.0e-4
lr_warmup = 1.0e-6
lr_decay_factor = 0.88
nonzero_patchave_threshold = -0.4
nonzero_pixel_threshold = -0.2
resume_optimizer_state = True
use_single_loader = False
#checkpoint_dir="/n/holystore01/LABS/iaifi_lab/Users/twongjirad/larmae_checkpoints/"
checkpoint_dir="/home/twongjirad/working/larbys/larmae/weights_trash"

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
    with open(args.config_file,'r') as f:
        cfg = yaml.safe_load(f)

    # set device
    torch.cuda.set_device(gpu)
    DEVICE = torch.device("cuda:%d"%(gpu) if torch.cuda.is_available() else "cpu")

    mae = load_model( args.config_file, strict=True, remove_ddp_prefix=True ).to(DEVICE)
    # v = ViT(
    #     image_size = 512,
    #     channels = 1,
    #     patch_size = 16,
    #     num_classes = 5,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 8,
    #     mlp_dim = 2048
    # )

    # mae = MAE(
    #     encoder = v,
    #     masking_ratio = 0.50,
    #     decoder_dim = 512,
    #     decoder_depth = 6,
    #     weight_by_pixel=True,
    #     nonzero_pixel_threshold=nonzero_pixel_threshold
    # ).to(DEVICE)

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

    if not use_single_loader:
        loader = relarmaeMultiProcessDataloader(args.config_file, rank,
                                                num_workers=num_workers,
                                                prefetch_batches=1)
        nentries = len(loader)
        
    else:
        dataset = larmaeDataset( args.config_file, seed=rank )
        print("RANK[%d] dataset: "%(rank),dataset)
        print("pause for ROOT")

        loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                             shuffle=False,
                                             worker_init_fn = lambda id: np.random.seed(id),
                                             collate_fn=larmaeDataset.collate_fn)
        nentries = len(dataset)
    

    iters_per_epoch = int(nentries/(batch_size*args.gpus))
    
    print("RANK-%d Start Data Loader. Number of entries: "%(rank),nentries)
    print("RANK-%d Iters per epoch: "%(rank),iters_per_epoch)
    
    lr = LR
    optimizer = torch.optim.AdamW(mae.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)

    if resume_optimizer_state:
        ckptfile = cfg.get("model").get("checkpoint_file",None)
        if ckptfile is not None:
            print("Load optimizer state from checkpoint file: ",ckptfile)
            checkpoint_data = torch.load( ckptfile )
            optimizer.load_state_dict( checkpoint_data["optimizer"] )
        else:
            raise ValueError("Could not load checkpoint file to resume optimizer state")


    if rank==0 and LOG_WANDB:
        wandb.watch( mae, log='all', log_freq=NITERS_PER_MODEL_LOG )
    
    start = time.time()
    for iiter in range(START_ITER,START_ITER+NITERS):

        epoch = float(iiter)/float(iters_per_epoch)
        
        mae.train()
        optimizer.zero_grad(set_to_none=True)
        if rank==0:
            print("====================================")
            print("ITER[%d]"%(iiter))
    
        batch = next(iter(loader))
        imgs = batch["img"].to(DEVICE)
        #print("imgs: ",imgs.shape)
    
        maeloss, pred_masked, true_masked, masked_indices = mae(imgs,return_outputs=True)
        #print(masked_indices.shape)
        #print(masked_indices)
        #print(pred_masked.shape)
        #print(true_masked.shape)
    
        #print("Rank[",rank,"] Num nonzero patches: ",batch["num_nonzero_patches"])
        #print("Rank[",rank,"] Entries: ",batch["entry"])
        if rank==0:
            print("Rank[",rank,"] MAE-loss: ",maeloss)
        maeloss.backward()
        optimizer.step()

        # set the next learning rate
        lr = get_learning_rate_cosine_anealing_w_warmup( epoch, Tbase, warmup_epochs,
                                                         lr_min, lr_max, lr_warmup,
                                                         lr_decay_factor )
        #lr = 2.0e-5

        # update the optimizer
        for g in optimizer.param_groups:
            g['lr'] = lr
                
        if iiter%NITERS_PER_LOG==0:
            # Log information about this iteration
            logged = {}
            
            # accuracy calculations
            acc = {}
            with torch.no_grad():
                acc_dict1 = calc_zero_vs_nonzero_accuracy( pred_masked.detach(), 
                                                           true_masked.detach(),
                                                           nonzero_patchave_threshold )
                acc.update(acc_dict1)
            
                acc_dict2 = calc_occupied_pixel_accuracies( pred_masked.detach(), 
                                                            true_masked.detach(),
                                                            occupied_threshold=nonzero_pixel_threshold )
                acc.update(acc_dict2)

            # grab items into 
            for x in logged_list:
                logged[x] = acc[x]

            # add the MSE loss
            logged["loss"] = maeloss.detach().cpu().item()

            # log the learning rate
            logged["lr"] = lr

            # log the epoch
            logged["epoch"] = epoch

            # pass to wandb server
            if LOG_WANDB and rank==0:
                wandb.log( logged, step=int(iiter/NITERS_PER_LOG) )

            # dump memory usage
            if rank==0:            
                print(loader.monitor.table())

        # check pointing
        if rank==0 and iiter>START_ITER and iiter%NITERS_PER_CHECKPOINT==0:
            print("RANK-%d: Save Checkpoint"%(rank))
            save_checkpoint( {"iter":iiter,
                              "state_mae":mae.state_dict(),
                              "optimizer":optimizer.state_dict()},
                             False, iiter, tag="larmae", 
                             outdir=checkpoint_dir )

        if iiter%NITERS_PER_DATALOADER_RESET==0 and iiter>START_ITER:
            print("RANK[%d] reset dataloader"%(iiter))
            del loader
            nresets = int(iiter/NITERS_PER_DATALOADER_RESET)
            loader = relarmaeMultiProcessDataloader(args.config_file, (numresets+1)*(num_workers*args.gpus)+rank,
                                                    num_workers=num_workers,
                                                    prefetch_batches=1)
            

        # hail mary
        gc.collect()
    
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(NITERS)
    print("sec per iter: ",sec_per_iter)

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
    
