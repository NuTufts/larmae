import os,time,copy,sys
import torch
#import multiprocessing
import torch.multiprocessing as mp
import queue
from itertools import cycle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from larmae_dataset import larmaeDataset
import gc

from psutil import Process

def worker_fn(data_loader_config, index_queue, output_queue, rank, worker_idx, batch_size):
    # no dictionary

    seed = (rank+1)*(worker_idx+1)
    shuffle = False
    #dataset = {worker_idx: larmaeDataset( data_loader_config, seed=seed )}
    #print("worker[%d] dataset: "%(worker_idx),dataset[worker_idx])    
    dataset = larmaeDataset( data_loader_config, seed=seed )
    print("worker[%d] dataset: "%(worker_idx),dataset)

    torch.manual_seed( seed )
    
    #loader = {worker_idx: torch.utils.data.DataLoader(dataset[worker_idx],
    #                                                  batch_size=batch_size,
    #                                                  worker_init_fn = lambda id: np.random.seed(id+seed),
    #                                                  shuffle=shuffle)}

    loader = torch.utils.data.DataLoader(dataset,
                                         #num_workers=2,
                                         batch_size=batch_size,
                                         worker_init_fn = lambda id: np.random.seed(id+seed),
                                         collate_fn=larmaeDataset.collate_fn,
                                         shuffle=shuffle)
    # internal batch queue
    #worker_index_queue = [] # this list causing mem leaks?

    # this is so stupid
    current_buffer = -1
    worker_buffer_0 = None
    worker_buffer_1 = None
    worker_buffer_2 = None
    
    while True:
        # Worker function loop, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        
        try:
            # check for request for data
            #print("worker[",worker_idx,"] check request queue")
            index = index_queue.get(timeout=0.01)
            #index = index_queue.get()

            if index is None:
                print("worker[%d] saw index=None. Stop worker."%(worker_idx))
                #sys.stdout.flush()
                break
            
            if current_buffer>=0:
                #print("worker[",worker_idx,"] index[",index,"] using internal queue. buffer=",current_buffer)
                if current_buffer==0:
                    x = worker_buffer_0
                elif current_buffer==1:
                    x = worker_buffer_1
                elif current_buffer==2:
                    x = worker_buffer_2
            else:
                #print("worker[",worker_idx,"] internal queue is empty, get data from loader, idx=",index)
                #x = next(iter(loader[worker_idx]))
                x = next(iter(loader))

            #print("worker[%d] queueing index="%(worker_idx),index," with loader=",loader[worker_idx])                
            output_queue.put((index,x))

            if current_buffer>=0:
                # replace buffer with None and deref last ref x
                if current_buffer==0:
                    worker_buffer_0 = None
                elif current_buffer==1:
                    worker_buffer_1 = None
                elif current_buffer==2:
                    worker_buffer_2 = None
                del x
                current_buffer -= 1

            gc.collect()
            #print("worker process mem usage: %0.3f GB"%(Process().memory_info().rss/1.0e9))
            #print("gc: count=",gc.get_count())
            #sys.stdout.flush()
            continue
            
        except queue.Empty:
            #print("no request. fill worker queue")
            #if len(worker_index_queue)<batch_size:
            if current_buffer<2:
                #x = next(iter(loader[worker_idx]))
                current_buffer += 1
                if current_buffer==0:
                    worker_buffer_0 = next(iter(loader))
                elif current_buffer==1:
                    worker_buffer_1 = next(iter(loader))
                elif current_buffer==2:
                    worker_buffer_2 = next(iter(loader))

                print("fill worker[",worker_idx,"] buffer idx=",current_buffer)
                print("worker process mem usage: %0.3f GB"%(Process().memory_info().rss/1.0e9))
                #print("gc: count=",gc.get_count())                
                #sys.stdout.flush()
            #print("worker queue len=",len(worker_index_queue))

            continue


class larmaeMultiProcessDataloader():
    def __init__(self, data_loader_config, rank, batch_size,
                 num_workers=4,
                 prefetch_batches=3):

        self.index = 0
        self.batch_size = batch_size
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
        self.shuffle = True
        self.data_keys = None

        self.data_loader_config = data_loader_config
        self.dataset = larmaeDataset( data_loader_config )
                
        for iworker in range(num_workers):
            index_queue = mp.Queue()
            worker = mp.Process(
                target=worker_fn, args=(self.data_loader_config, index_queue, self.output_queue, rank, iworker, self.batch_size)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()
        print("prefetch finished")

    def prefetch(self):
        #print("prefetch")
        #while (self.prefetch_index < self.batch_size):
        #    # if the prefetch_index hasn't reached the end of the dataset
        #    # and it is not 2 batches ahead, add indexes to the index queues
        #    self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
        #    self.prefetch_index += 1
        self.index_queues[next(self.worker_cycle)].put(0)

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def __next__(self):
        #if self.index >= self.nentries:
        #    raise StopIteration
        #print("next")
        #out = [self.get() for _ in range(self.batch_size)]
        #out = [self.get() for _ in range(1)]
        #return self.collate_fn(out)
        out = self.get()
        return out

    def __len__(self):
        return len(self.dataset)

    def get(self):
        #print("start prefetch")
        #self.prefetch()
        #print("check cache")        
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            #print("check queue")            
            while True:
                try:
                    (index, data) = self.output_queue.get()
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
                w.join(timeout=1000.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()

    def collate_fn(self,batch):
        if self.data_keys is None:
            self.data_keys = [k for k in batch[0].keys()]
            print("set data keys: ",self.data_keys)
            
        out = {}
        for b in self.data_keys:
            if type(batch[0][b]) is torch.Tensor:
                s = list(batch[0][b].shape)
                #print("shape: ",s)
                s[0] = len(batch)
                x = torch.zeros( s )
                for i,idata in enumerate(batch):
                    #print("check: [",b,"][",i,"]: sum=",idata[b].sum())
                    x[i] = idata[b]
                out[b] = x
            
        return out
        
if __name__ == "__main__":
    
    import yaml
    from larcv import larcv
    larcv.load_pyutil()
    
    config = "config_train.yaml"
    FAKE_NET_RUNTIME = 0.250
    niters = 10
    batch_size = 64
    num_workers = 2
    rank = 0
    loader = larmaeMultiProcessDataloader(config, rank, batch_size,
                                          num_workers=num_workers,
                                          prefetch_batches=1)

    print("START LOOP")
    print("[enter] to continue")
    input()

    start = time.time()
    dt_load = 0.0
    for iiter in range(niters):
        print("-------------------")
        print("ITER ",iiter)
        dt_iter = time.time()
        batch = next(iter(loader))
        dt_iter = time.time()-dt_iter
        dt_load += dt_iter

        print("main process mem usage: %0.3f GB"%(Process().memory_info().rss/1.0e9))        
        if FAKE_NET_RUNTIME>0:
            print("pretend network: lag=",FAKE_NET_RUNTIME)
            time.sleep( FAKE_NET_RUNTIME )
        
        # Dump data for debug
        print("entry: ",batch["entry"])
        if False:
            print("batch: ",batch)
            print(batch["entry"])
            for ib in range(batch["img"].shape[0]):
                print("img[",ib,"] check: ",batch["img"][ib,:].sum())
                print("nonzero dump: ")
                x = batch["img"][ib].reshape(-1)
                print(x[x>-0.4][:10])
                
    print("MADE IT")
    print("loading time per iteration: ",dt_load/float(niters))
    end = time.time()
    print("WALL time per iter: ",(end-start)/float(niters))
