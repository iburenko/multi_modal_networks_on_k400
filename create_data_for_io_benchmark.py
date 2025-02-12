import os
from os import path
import multiprocessing

from tqdm import tqdm
import torch
import torchvision as tv

def create_data(worker_id, batch_size, save_to_dir, num_samples = 2**12):
    dataset_dir = path.join(save_to_dir, "bs" + str(batch_size))
    video_fps = 30
    snippet_len = 10
    num_iters = 64 // batch_size
    for i in range(num_iters):
        sample_id = num_iters * worker_id + i
        video = torch.randint(0, 256, (batch_size * video_fps * snippet_len, 320, 320, 3)).type(torch.uint8)
        video_fp = path.join(dataset_dir, "sample_" + str(sample_id ).zfill(5) + ".mp4")
        tv.io.write_video(video_fp, video, video_fps, video_codec="h264")
        # print("Worker #", worker_id, ". Global fid = ", sample_id, flush=True)

def main():
    job_id = os.environ.get("SLURM_JOB_ID", "default_job")
    hostname = "alex"
    save_to_dir = path.join("/tmp", f"{job_id}.{hostname}", "data")

    batch_sizes = [2**i for i in [4,5]]
    num_cpus = 64

    for bs in batch_sizes:
        tasks = [(worker_id, bs, save_to_dir) for worker_id in range(num_cpus)]
        with multiprocessing.Pool(processes=num_cpus) as pool:
            pool.starmap(create_data, tasks)

if __name__ == "__main__":
    main()
