from os import path, environ, listdir as ls
from time import time

import numpy as np
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision as tv

class BenchmarkDataset(Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.hostname = "alex"
        job_id = environ["SLURM_JOB_ID"]
        dataset_path = f"/tmp/{job_id}.{self.hostname}/data/bs{self.batch_size}"
        all_files = sorted(ls(dataset_path))
        self.all_fps = [path.join(dataset_path, elem) for elem in all_files]

    def __len__(self):
        return len(self.all_fps)
    
    def __getitem__(self, idx):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        video, *_ = tv.io.read_video(self.all_fps[idx], pts_unit="sec")
        # print("Worker", worker_id, "File id", idx)
        return video

def benchmark(batch_size):
    dataset = BenchmarkDataset(batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=32 // batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        )
    batch_loading_time = np.zeros(len(dataloader))
    global_start_time = time()
    start_time = time()
    for i, batch in enumerate(dataloader):
        batch_loading_time[i] = time() - start_time
        start_time = time()
    final_time = time() - global_start_time
    print("batch size", batch_size)
    print("final time", final_time)
    print("final time batch-wise", batch_loading_time.sum())
    print("mean batch time", batch_loading_time.mean())
    print("std batch time", batch_loading_time.std())

def main():
    for batch_size in [2**i for i in range(4)]:
        benchmark(batch_size)

if __name__ == "__main__":
    main()