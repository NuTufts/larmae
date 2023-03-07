import os,time,copy,sys
#import multiprocessing
import torch.multiprocessing as mp
import queue
from itertools import cycle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from larmae_dataset import larmaeDataset

def worker_fn(dataset, index_queue, output_queue, worker_idx):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=60)
        except queue.Empty:
            continue
        if index is None:
            print("worker[%d] saw index=None"%(worker_idx))
            break
        #print("worker[%d] queueing index="%(worker_idx),index," with loader=",dataset)
        x = dataset[index]
        #print("  ",x.keys())
        output_queue.put((index,x))


class larmaeMultiProcessDataloader():
    def __init__(self, data_loader_config, batch_size,
                 num_workers=4,
                 prefetch_batches=3,
                 collate_fn=larmaeDataset.collate_fn ):

        self.index = 0
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = mp.Queue()
        self.index_queues = []
        self.workers = []
        self.loaders = []
        self.worker_cycle = cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0
        self.nentries = 0

        self.data_loader_config = data_loader_config
        
        num_triplet_samples = None
        if "NUM_TRIPLET_SAMPLES" in data_loader_config:
            num_triplet_samples = data_loader_config["NUM_TRIPLET_SAMPLES"]
            print("NUM TRIPLET SAMPLES: ",num_triplet_samples)
        
        for iworker in range(num_workers):
            loader = larmaeDataset( txtfile=data_loader_config["INPUT_FILE"],
                                      random_access=data_loader_config["RANDOM_ACCESS"],
                                      sequential_access=False,                                      
                                      num_triplet_samples=num_triplet_samples,
                                      triplet_limit=data_loader_config["TRIPLET_LIMIT"],
                                      return_constant_sample=data_loader_config["CONSTANT_SAMPLE"],
                                      return_triplet_planecharge=data_loader_config["RETURN_TRIPLET_PLANECHARGE"],
                                      use_keypoint_sampler=data_loader_config["USE_KEYPOINT_SAMPLER"],
                                      use_qcut_sampler=data_loader_config["USE_QCUT_SAMPLER"],
                                      load_truth=data_loader_config["LOAD_TRUTH"],
                                      verbose=data_loader_config["VERBOSE"] )
            print("worker[%d] loader: "%(iworker),loader)
            if iworker==0:
                self.nentries = len(loader)                
            self.loaders.append(loader)
            index_queue = mp.Queue()
            worker = mp.Process(
                target=worker_fn, args=(loader, index_queue, self.output_queue, iworker)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self):
        while (self.prefetch_index < self.prefetch_batches*self.batch_size):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1            

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def __next__(self):
        if self.index >= self.nentries:
            raise StopIteration
        out = self.collate_fn([self.get() for _ in range(self.batch_size)])        
        return out

    def get(self):
        #print("start prefetch")
        self.prefetch()
        #print("check cache")        
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            #print("check queue")            
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=60)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data
        #print("increment index")            
        self.index += 1
        return item

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
        
if __name__ == "__main__":
    
    import yaml
    stream = open("config/test_loader.yaml", 'r')
    toplevel_config = yaml.load(stream, Loader=yaml.FullLoader)
    config = toplevel_config["TRAIN_DATALOADER_CONFIG"]
    
    FAKE_NET_RUNTIME = 1.0
    niters = 10
    
    loader = larmaeMultiProcessDataloader(config,4,num_workers=2,prefetch_batches=1,
                                            collate_fn=larmaeDataset.collate_fn_singletensor)
    print("START LOOP")
    print("[enter] to continue")
    input()

    dt_load = 0.0
    for iiter in range(niters):
        print("-------------------")
        print("ITER ",iiter)
        dt_iter = time.time()
        batch = next(iter(loader))
        dt_iter = time.time()-dt_iter
        dt_load += dt_iter
        print("batch keys: ",batch.keys())
        print("tree entries: ",batch["tree_entry"])
        for k in batch.keys():
            if type(batch[k]) is torch.Tensor:
                print("  ",k,": ",batch[k].shape)
        print("pretend network: lag=",FAKE_NET_RUNTIME)
        time.sleep( FAKE_NET_RUNTIME )
    print("MADE IT")
    print("loading time per iteration: ",dt_load/float(niters))
