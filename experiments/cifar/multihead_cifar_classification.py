from argparse import ArgumentParser
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random
import warnings
from glob import glob

import torch
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np


@dataclass
class Adaptor(torch.nn.Module):
    input_dims: list[int]
    num_classes: int
    num_heads: Optional[int] = None
    hidden_dim: int = 1000
    
    def __post_init__(self):
        super().__init__()
        self.adaptor_input_dim = sum(self.input_dims)
        self.adaptor = self.create_adaptor()

    def create_adaptor(self):
        return torch.nn.Identity()
        adaptor = torch.nn.Sequential(
            torch.nn.Linear(self.adaptor_input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.num_classes)
        )
        return adaptor
    
    def forward(self, x):
        return self.adaptor(x)


@dataclass
class MultiHeadCNN(torch.nn.Module):
    model_name: str | list[str]
    num_classes: int
    num_heads: Optional[int] = None
    copy_heads: Optional[bool] = None

    def __post_init__(self):
        super().__init__()
        # self.models_name_list = self.create_models_name_list()
        self.model = self.create_model()

    def create_model(self):
        return timm.create_model(self.model_name, pretrained=False, num_classes=10)
        features_dim_list = self._convert_model_names_to_feat_dims()
        self.heads = [
            timm.create_model(model_name, num_classes=0)
            for model_name in self.models_name_list
        ]
        if (
            self.copy_heads is not None # copy_heads = True
            and len(self.models_name_list) > 1 # We have more than one head
            and len(set(self.models_name_list)) == 1 # all heads are equal
            ):
            self._copy_weights()
        else:
            warn_str = (
                "copy_heads = True will be ignored!\n"
                "copy_heads is True, but some heads have different architectures."
                "Either the list of the models is wrong, or copy_heads = True"
                "was provided by mistake.\n"
                "Please, check the parameters!"
            )
            warnings.warn(warn_str)
        self.adaptor = Adaptor(features_dim_list, self.num_classes)
        
    def create_models_name_list(self):
        models_name_list = None
        if isinstance(self.model_name, str):
            if self.num_heads is None or self.num_heads == 1:
                self.num_heads == 1
                models_name_list = [self.model_name]
            elif self.num_heads > 1:
                models_name_list = [self.model_name] * self.num_heads
        else:
            models_name_list = self.model_name
        if models_name_list is None:
            raise_str = (
                "The values of model_name and num_heads do not allow"
                "to produce a string of model names.\n"
                "Please, check the variables."
                )
            raise ValueError(raise_str)
        return models_name_list
    
    def _convert_model_names_to_feat_dims(self):
        features_dim_list = list()
        for model_name in self.models_name_list:
            if model_name == "resnet18":
                features_dim_list.append(512)
            elif model_name == "resnet34":
                features_dim_list.append(512)
            elif model_name == "resnet50":
                features_dim_list.append(2048)
            elif model_name == "resnet101":
                features_dim_list.append(2048)
            elif model_name == "resnet152":
                features_dim_list.append(2048)
        return features_dim_list
    
    def _copy_weights(self):
        main_head = self.heads[0]
        for head in self.heads[1:]:
            head.load_state_dict(main_head.state_dict())
        
    def forward(self, x):
        return self.model(x)
        features = torch.hstack([head(x) for head in self.heads])
        return self.adaptor(features)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(model, criterion, optimizer, train_dataloader):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for x, y in train_dataloader:
        model.zero_grad()
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    acc = correct / total
    return acc, train_loss

def test(model, criterion, test_dataloader):
    _ = model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    acc = correct / total
    return acc, test_loss

def save_checkpoint(checkpoints_dir, metric_name, metric_val, model, epoch, name_prefix):
    metric_val = round(metric_val, 3)
    fn = "checkpoint_" + name_prefix + "_" + metric_name + "_" + str(metric_val) + "_epoch_" + str(epoch) + ".pth"
    fp = checkpoints_dir.joinpath(fn)
    print(fn)
    print(fp)
    state = {
            "model": model.state_dict(),
            "metric_name": metric_name,
            "metric_val": metric_val,
            "epoch": epoch,
        }
    torch.save(state, fp)
    return 0

def delete_old_checkpoint(checkpoints_dir, metric_name, name_prefix):
    search_pattern = checkpoints_dir.joinpath("checkpoint_" + name_prefix + "_" + metric_name + "*")
    for fp in glob(str(search_pattern)): 
        # fp = checkpoints_dir.joinpath(fn)
        os.remove(fp)
    return 0

def freeze_weights(model, submodel_name):
    if submodel_name == "":
        return model
    if submodel_name == "resnet26":
        for m in model.layer2[2].parameters():
            m.requires_grad = False
        for m in model.layer3[2:5].parameters():
            m.requires_grad = False
        return model
    if submodel_name == "resnet18":
        for m in model.layer1[1].parameters():
            m.required_grad = False
        for m in model.layer2[1:3].parameters():
            m.requires_grad = False
        for m in model.layer3[1:5].parameters():
            m.requires_grad = False
        for m in model.layer4[1].parameters():
            m.requires_grad = False
        return model
    
def create_name_prefix(submodel_name, model_name):
    if submodel_name == "":
        return "rn" + model_name[-2:]
    else:
        return "rn" + submodel_name[-2:] + "_in_" + model_name[-2:]
    
def get_data(dataset_name, run_dir):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=run_dir,
            train=True,
            download=True,
            transform=transform_train
            )
        testset = torchvision.datasets.CIFAR10(
            root=run_dir,
            train=False,
            download=True,
            transform=transform_test
            )
    elif dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=run_dir,
            train=True,
            download=True,
            transform=transform_train
            )
        testset = torchvision.datasets.CIFAR100(
            root=run_dir,
            train=False,
            download=True,
            transform=transform_test
            )
    return trainset, testset

def main(args):
    node_user_name = os.environ["USER"]
    node_slurm_job_id = os.environ["SLURM_JOB_ID"]
    datadir = node_user_name + "." + node_slurm_job_id
    # datadir = node_user_name
    run_dir = Path("/tmp").joinpath(datadir)
    print("Start!")
    # checkpoints_dir = Path("/projects/p_scads_nlp/ilbu282f/mm_landscape/experiments/checkpoints")
    checkpoints_dir = run_dir.joinpath("checkpoints")
    epochs = args.epochs
    model_name = args.model
    submodel_name = args.submodel
    dataset_name = args.dataset
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    print('create a model ', model_name)
    model = timm.create_model(model_name, num_classes=num_classes)
    print('finish')
    # model = freeze_weights(model, submodel_name)
    _ = model.cuda()
    print("Model is ready")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    print("optimizer and criterion are ready")
    g = torch.Generator()
    g.manual_seed(34)
    trainset, testset = get_data(dataset_name, run_dir)
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=12,
        worker_init_fn=seed_worker,
        generator=g
        )
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=512,
        shuffle=False,
        num_workers=12
        )
    print("dataloaders are ready")
    if args.multiple_head == True:
        checkpoints_dir = checkpoints_dir.joinpath("multiple_head")
    else:
        checkpoints_dir = checkpoints_dir.joinpath("single_head")
    best_acc = 0
    # best_loss = 100
    name_prefix = create_name_prefix(submodel_name, model_name)
    for epoch in range(epochs):
        print("Start epoch ", epoch)
        train_acc, train_loss = train(model, criterion, optimizer, train_dataloader)
        test_acc, test_loss = test(model, criterion, test_dataloader)
        if test_acc > best_acc:
            delete_old_checkpoint(checkpoints_dir, "acc", name_prefix)
            best_acc = test_acc
            save_checkpoint(checkpoints_dir, "acc", best_acc, model, epoch, name_prefix)
        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     save_checkpoint(checkpoints_dir, "loss", best_loss, model, epoch)
        scheduler.step()
        print(scheduler.get_last_lr())
        print("End epoch ", epoch)
        print("Best acc = ", best_acc)
        # print("Best loss = ", best_loss)
        print("="*80)


if __name__ == "__main__":
    torch.manual_seed(34)
    random.seed(34)
    np.random.seed(34)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--submodel", type=str, default="")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--multiple-head", action="store_true")
    parser.add_argument("--copy-heads", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)