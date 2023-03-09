from __future__ import print_function
import os,sys,argparse,time
#sys.path.append("/usr/local/lib/python3.8/dist-packages/")
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")
from ctypes import c_int
import numpy as np

"""
Extract images larcv/larlite information from MicroBooNE into numpyarray products for easy loading
"""
parser = argparse.ArgumentParser("Make mlreco data from larcv")
parser.add_argument("-i","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tf",  "--tick-forward",action='store_true',default=False,help="Input LArCV data is tick-forwards [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
parser.add_argument("-e",   "--start-entry",type=int,default=0,help="Entry to start [default: 0]")
parser.add_argument("-d",   "--detector",type=str,default='uboone',help='Detector geometry configuration [default: "uboone"')
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
from ROOT import larutil


# SET DETECTOR
if args.detector not in ["uboone","sbnd","icarus"]:
    raise ValueError("Invalid detector. Choices: uboone, sbnd, or icarus")
    
if args.detector == "icarus":
    detid = larlite.geo.kICARUS
elif args.detector == "uboone":
    detid = larlite.geo.kMicroBooNE
elif args.detector == "sbnd":
    detid = larlite.geo.kSBND
larutil.LArUtilConfig.SetDetector(detid)

rt.gStyle.SetOptStat(0)

# OPEN INPUT FILES: LARCV
if args.tick_forward:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( args.input_larcv )
if not args.tick_forward:
    iolcv.reverse_all_products()
#iolcv.
iolcv.initialize()

# GET NUMBER OF ENTRIES: LARCV
nentries = iolcv.get_n_entries()
print("Number of entries in file: ",nentries)

# DID USER SPECIFY A START ENTRY?
start_entry = args.start_entry
if start_entry>=nentries:
    print("Asking to start after last entry in file")
    sys.exit(0)

# DID USER SPECIFY THE NUMBER OF ENTRIES TO PROCESS
if args.nentries>0:
    end_entry = start_entry + args.nentries
else:
    end_entry = start_entry + nentries
if end_entry>=nentries:
    end_entry = nentries

# CREATE THE OUTPUT FILE
outfile = rt.TFile(args.output,"recreate")
ttree = rt.TTree("extbnb_images","EXTBNB wireplane images")
rse_v  = rt.std.vector("int")()
meta_v = rt.std.vector("larcv::ImageMeta")()
img_v  = rt.std.vector("larcv::NumpyArrayFloat")()
ttree.Branch("meta_v",meta_v)
ttree.Branch("img_v",img_v)

# -------------------
# EVENT LOOP!!

start = time.time()
for ientry in range(start_entry,end_entry,1):

    print(" ") 
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    sys.stdout.flush()

    # clear containers
    rse_v.clear()
    meta_v.clear()
    img_v.clear()
    
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data(larcv.kProductImage2D,args.adc)
    adc_v = ev_adc.as_vector()
    if adc_v.size()!=3:
        raise ValueError("Number of images in this event is not 3?")
    
    for p in range(3):
        np_img = np.transpose( larcv.as_ndarray( adc_v.at(p) ) ).astype(np.float32)
        x = larcv.NumpyArrayFloat( np_img )
        img_v.push_back( x )
        meta_v.push_back( adc_v.at(p).meta() )
    
    # Done with the event -- Fill it!
    #print("save entry")
    rse_v.push_back( ev_adc.run() )
    rse_v.push_back( ev_adc.subrun() )
    rse_v.push_back( ev_adc.event() )
    
    ttree.Fill()

print("Event Loop Finished")
print("Writing Output File")
ttree.Write()
print("Done")
outfile.Close()

iolcv.finalize()

