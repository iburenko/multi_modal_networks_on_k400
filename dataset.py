import json
import random
from pathlib import Path

import torch
import torchvision as tv
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import Dataset
import numpy as np
import librosa

class KineticsDataset(Dataset):
    def __init__(self, split, modality, T=16, crop_size=224):
        json_home = Path("/data/horse/ws/ilbu282f-mm_landscape/kinetics-dataset/k400/jsons")
        self.split = split
        self.modality = modality
        self.T = T
        self.crop_size = crop_size
        json_fn = json_home.joinpath("final_" + self.split + "_dict_v3.json")
        data = json.load(open(json_fn))
        self.data = {key: val for key, val in list(data.items())}
        all_labels = sorted(set(elem[1] for elem in self.data.values()))
        self.label_dict = {label: i for i, label in enumerate(all_labels)}
        self.train_transforms = [
            transforms_v2.RandomCrop(self.crop_size),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ]
        self.val_transforms = [
            transforms_v2.CenterCrop(self.crop_size),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ]
        self.resized_min_size = 256
        self.resized_max_size = 320
        self.val_num_sampler_per_video = 10

    def __len__(self):
        if self.split == "train":
            return 2**20
            return 4 * len(self.data)
        else:
            # return 2**9
            return len(self.data)
        
    def _get_resize_transform(self, metadata):
        video_height = metadata["video_shape"][2]
        video_width = metadata["video_shape"][3]
        if self.split == "train":
            min_size = random.randint(self.resized_min_size, self.resized_max_size)
        else:
            min_size = self.resized_min_size
        scale = min_size / min(video_height, video_width)
        max_size = int(scale * max(video_height, video_width))
        if video_height < video_width:
            size_tuple = (min_size, max_size)
        else:
            size_tuple = (max_size, min_size)
        return transforms_v2.Resize(size_tuple, antialias=True)
    
    def apply_transforms(self, snippet, metadata):
        resize_transform = self._get_resize_transform(metadata)
        if self.split == "train":
            transforms = tv.transforms.Compose(
                [resize_transform] + self.train_transforms
                )
        else:
            transforms = tv.transforms.Compose(
                [resize_transform] + self.val_transforms
                )
        return transforms(snippet)
    
    def _align_audio_with_snippet(self, audio, start_frame, video_fps, audio_fps):
        snippet_len_sec = self.T / video_fps
        audio_frames = int(audio_fps * snippet_len_sec)
        audio_start_frame = int(audio_fps * (start_frame / video_fps))
        return audio[:, audio_start_frame:audio_start_frame + audio_frames]
    
    def generate_log_mel_spectrogram(self, audio, audio_fps):
        if audio.shape[0] == 2:
            audio = audio.mean(axis=0)
        elif audio.shape[0] == 1:
            audio = audio[0]
        n_mels = 40
        n_fft = 2048
        hop_length = len(audio) // 100
        if hop_length > 0:
            mel_spec = librosa.feature.melspectrogram(
                y=audio.numpy(), 
                sr=audio_fps, 
                n_mels=n_mels, 
                n_fft=n_fft, 
                hop_length=hop_length
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)[:, :100]
        else:
            log_mel_spec = np.zeros((40, 100)).astype(np.float32)
        return torch.from_numpy(log_mel_spec)

    def __getitem__(self, idx):
        if self.split == "train":
            idx %= len(self.data)
        video_fp, label, metadata = self.data[str(idx)]
        video, audio, _ = tv.io.read_video(video_fp, pts_unit="sec", output_format="TCHW")
        video_len = metadata["video_shape"][0]
        video_fps = metadata["video_fps"]
        audio_fps = metadata["audio_fps"]
        if self.split == "train":
            start_ind = random.randint(0, video_len - self.T)
            audio_snippet = self._align_audio_with_snippet(audio, start_ind, video_fps, audio_fps)
            audio_features = self.generate_log_mel_spectrogram(audio_snippet, audio_fps)
            if self.modality == "audio":
                video_snippet = torch.zeros(self.T, 3, self.crop_size, self.crop_size)
            else:
                video_snippet = video[start_ind:start_ind + self.T]
                video_snippet = self.apply_transforms(video_snippet, metadata)
        else:
            start_inds = torch.linspace(0, video_len - self.T, self.val_num_sampler_per_video).int()
            video_snippet = torch.zeros(self.val_num_sampler_per_video, self.T, 3, self.crop_size, self.crop_size)
            audio_features = torch.zeros(self.val_num_sampler_per_video, 40, 100)
            for i, start_ind in enumerate(start_inds):
                audio_sub_snippet = self._align_audio_with_snippet(audio, start_ind, video_fps, audio_fps)
                audio_sub_snippet_feats = self.generate_log_mel_spectrogram(audio_sub_snippet, audio_fps)
                audio_features[i] = audio_sub_snippet_feats
                if self.modality == "audio":
                    continue
                else:
                    video_sub_snippet = video[start_ind:start_ind + self.T]
                    video_snippet[i] = self.apply_transforms(video_sub_snippet, metadata)
        return video_snippet, audio_features, self.label_dict[label]