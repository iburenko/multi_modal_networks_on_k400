from os import listdir as ls
from datetime import datetime


def print_current_time():
    print(datetime.now().strftime("%H:%M:%S %Y-%m-%d"), flush=True)

def get_last_checkpoint(ckpt_path):
    model_name = sorted([elem for elem in ls(ckpt_path) if "val_acc" in elem])[-1]
    print("We use the last checkpoint",model_name, flush=True)
    return ckpt_path.joinpath(model_name)

def get_experiment_number(ckpt_path):
    all_experiments = sorted(ls(ckpt_path))
    if len(all_experiments) == 0:
        experiment_num = str(0).zfill(4)
    else:
        last_experiment = all_experiments[-1]
        last_experiment_num = int(last_experiment.split("_")[-1])
        experiment_num = str(last_experiment_num + 1).zfill(4)
    return experiment_num

def get_experiment_folder_name(job_id, master_job_id, ckpt_modality_model_path):
    if master_job_id == 0:
        experiment_number = get_experiment_number(ckpt_modality_model_path)
        folder_name = f"experiment_{job_id}_{experiment_number}"
    else:
        folder_name = [elem for elem in ls(ckpt_modality_model_path) if str(master_job_id) in elem][0]
    return folder_name
