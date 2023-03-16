import os,sys
import torch

def calc_zero_vs_nonzero_accuracy(pred_patches,true_patches,nonzero_ave_threshold):
    """
    pred_patches (num batches,num masked patches,patch height*width)
    """

    b,n,m=true_patches.shape
    
    # get sum of patches
    patch_sum = torch.sum( true_patches, dim=2 )/float(m) # (B,P)
    print("patch_sum: ",patch_sum.shape)

    # mask of patches with zero sum: true empty
    zero_sum_mask = (patch_sum<=nonzero_ave_threshold).reshape( -1 )
    print("zero_sum_mask: ",zero_sum_mask.shape," sum=",zero_sum_mask.sum())
    if zero_sum_mask.sum()>0:

        zero_patches_pred = pred_patches.reshape( b*n, m)[ zero_sum_mask[:] ]
        print("zero_patches_pred: ",zero_patches_pred.shape)
        
        zero_patches_true = true_patches.reshape( b*n, m )[ zero_sum_mask[:] ]
        print("zero_patches_true: ",zero_patches_true.shape)
        
        # average mean-squared error: basically, does it predict zero properly
        mse_zero = torch.nn.functional.mse_loss( zero_patches_true, zero_patches_pred ).cpu().item()

    else:
        mse_zero = 0.0

    # mask of patches with zero sum: true empty
    nonzero_sum_mask = (patch_sum>nonzero_ave_threshold).reshape(-1)

    if nonzero_sum_mask.sum()>0:

        nonzero_patches_pred = pred_patches.reshape( b*n, m)[ nonzero_sum_mask[:] ]
        print(nonzero_patches_pred.shape)

        nonzero_patches_true = true_patches.reshape( b*n, m)[ nonzero_sum_mask[:] ]
    
        # average mean-squared error: basically, does it predict zero properly
        mse_nonzero = torch.nn.functional.mse_loss( nonzero_patches_true, nonzero_patches_pred ).cpu().item()

    else:

        mse_nonzero = 0.0
        
    
    return {"mse_zero":mse_zero, "mse_nonzero":mse_nonzero}

def calc_occupied_pixel_accuracies(pred_patches,true_patches, occupied_threshold=-0.8):

    occupied_mask = (true_patches>=occupied_threshold).reshape(-1)
    zero_mask     = (true_patches<occupied_threshold).reshape(-1)    

    zero2zero     = ( pred_patches.reshape(-1)[ zero_mask ] < occupied_threshold ).sum()
    zero2occupied = ( pred_patches.reshape(-1)[ zero_mask ] >= occupied_threshold ).sum()
    occupied2occupied = ( pred_patches.reshape(-1)[ occupied_mask ] >= occupied_threshold ).sum()
    occupied2zero     = ( pred_patches.reshape(-1)[ occupied_mask ] < occupied_threshold ).sum()

    acc = {}
    acc["num_zero"]  = zero_mask.sum().cpu().item()    
    if zero_mask.sum()>0:
        acc["zero2zero"]     = zero2zero.cpu().item()/float(zero_mask.sum().cpu().item())
        acc["zero2occupied"] = zero2occupied.cpu().item()/float(zero_mask.sum().cpu().item())
    else:
        acc["zero2zero"]     = None
        acc["zero2occupied"] = None
        

    acc["num_occupied"]  = occupied_mask.sum().cpu().item()
    if occupied_mask.sum()>0:
        acc["occupied2zero"]     = occupied2zero.cpu().item()/float(occupied_mask.sum().cpu().item())
        acc["occupied2occupied"] = occupied2occupied.cpu().item()/float(occupied_mask.sum().cpu().item())
    else:
        acc["zero2zero"]     = None
        acc["zero2occupied"] = None

    return acc
        

def calc_accuracies():
    pass
