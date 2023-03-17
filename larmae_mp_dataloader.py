import os,time,copy,sys
import torch
#import multiprocessing
import torch.multiprocessing as mp
import queue
from itertools import cycle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from larmae_dataset import larmaeDataset
#import samplers

def worker_fn(data_loader_config, index_queue, output_queue, rank, worker_idx, batch_size):

    seed = (rank+1)*(worker_idx+1)
    shuffle = False
    dataset = {worker_idx: larmaeDataset( data_loader_config, seed=seed )}
    print("worker[%d] dataset: "%(worker_idx),dataset[worker_idx])

    torch.manual_seed( seed )
    #sampler = torch.utils.data.SequentialSampler( dataset )
    #sampler = samplers.RandomSequenceSampler.create( len(dataset), batch_size, seed=worker_idx )
    
    loader = {worker_idx: torch.utils.data.DataLoader(dataset[worker_idx],
                                                      #num_workers=1,
                                                      #persistent_workers=True,
                                                      #sampler=sampler,
                                                      batch_size=batch_size,
                                                      worker_init_fn = lambda id: np.random.seed(id+seed),
                                                      shuffle=shuffle)}
    # internal batch queue
    worker_index_queue = []
    
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
            
            if len(worker_index_queue)>0:
                print("worker[",worker_idx,"] index[",index,"] using internal queue. len=",len(worker_index_queue))
                x = worker_index_queue.pop()
            else:
                print("worker[",worker_idx,"] internal queue is empty, get data from loader, idx=",index)
                x = next(iter(loader[worker_idx]))

            #print("worker[%d] queueing index="%(worker_idx),index," with loader=",loader[worker_idx])                
            output_queue.put((index,x))
            #sys.stdout.flush()
            continue
            
        except queue.Empty:
            #print("no request. fill worker queue")
            #if len(worker_index_queue)<batch_size:
            if len(worker_index_queue)<10:
                x = next(iter(loader[worker_idx]))
                worker_index_queue.append(x)
                #print("fill worker[",worker_idx,"] queue. len=",len(worker_index_queue))
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
        print("prefetch")
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
        print("next")
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
    
    config = "config_train.yaml"
    FAKE_NET_RUNTIME = 1.0
    niters = 10
    batch_size = 1
    num_workers = 1
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

        # Dump data for debug
        print("entry: ",batch["entry"])
        if True:
            print("batch: ",batch)
            print(batch["entry"])
            for ib in range(batch["img"].shape[0]):
                print("img[",ib,"] check: ",batch["img"][ib,:].sum())
                print("nonzero dump: ")
                x = batch["img"][ib].reshape(-1)
                print(x[x>-0.4][:10])
                
        if FAKE_NET_RUNTIME>0:
            print("pretend network: lag=",FAKE_NET_RUNTIME)
            time.sleep( FAKE_NET_RUNTIME )
    print("MADE IT")
    print("loading time per iteration: ",dt_load/float(niters))
    end = time.time()
    print("WALL time per iter: ",(end-start)/float(niters))
