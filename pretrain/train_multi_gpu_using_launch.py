import os
import math
import tempfile
import argparse
from xmlrpc.client import boolean

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from torchvision import datasets, transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
from model import Cifarmodel as VitCifarmodel
from model import Mnistmodel as VitMnistmodel
from swin import Cifarmodel as SwinCifarmodel
# from SpikingSwinTransformer import Cifarmodel as SwinCifarmodel
# from SpikingSwinTransformer import Mnistmodel as SwinMnistmodel
import wandb
import matplotlib.pyplot as plt
# from torchsummary import summary
from timm.scheduler.cosine_lr import CosineLRScheduler

class WarmupLR:
 def __init__(self, optimizer, num_warm) -> None:
  self.optimizer = optimizer
  self.num_warm = num_warm
  self.lr = [group['lr'] for group in self.optimizer.param_groups]
  self.num_step = 0
 
 def __compute(self, lr) -> float:
  return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))
 
 def step(self) -> None:
  self.num_step += 1
  lr = [self.__compute(lr) for lr in self.lr]
  for i, group in enumerate(self.optimizer.param_groups):
      group['lr'] = lr[i]




def getdataset(args):
    data_path = args.data_path
    device = torch.device(args.device)
    size = args.size
    if args.dataset == 'imagenet':
        train_path = os.path.join(data_path, 'train/ILSVRC2012_img_train')
        val_path = os.path.join(data_path, 'val')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_data_set = datasets.ImageFolder(root=train_path, transform=transform_train)
        # total = len(train_data_set)
        # train_data_set, _= torch.utils.data.random_split(train_data_set, [40000, total-40000])

        val_data_set = datasets.ImageFolder(root=val_path, transform=transform_test)

        
        if args.model == 'swin':
            model = SwinCifarmodel(1000).to(device)
        elif args.model == 'vit':
            model = VitCifarmodel().to(device)
    if args.dataset == 'Cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data_set = torchvision.datasets.CIFAR100(data_path, train=True,
                                                      download=True, transform=transform_train)

        val_data_set = torchvision.datasets.CIFAR100(data_path, train=False,
                                                    download=True, transform=transform_test)

        
        if args.model == 'swin':
            model = SwinCifarmodel(100).to(device)
        elif args.model == 'vit':
            model = VitCifarmodel().to(device)
    if args.dataset == 'Cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data_set = torchvision.datasets.CIFAR10(data_path, train=True,
                                                      download=True, transform=transform_train)

        val_data_set = torchvision.datasets.CIFAR10(data_path, train=False,
                                                    download=True, transform=transform_test)

        
        if args.model == 'swin':
            model = SwinCifarmodel().to(device)
        elif args.model == 'vit':
            model = VitCifarmodel().to(device)
    elif args.dataset == 'Mnist':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

        train_data_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        val_data_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        if args.model == 'swin':
            model = VitMnistmodel().to(device)
        elif args.model == 'vit':
            model = VitMnistmodel().to(device)
    elif args.dataset == 'catsdogs':
        tr = "train"
        te = "test"
        transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_data_set = datasets.ImageFolder(os.path.join(data_path, tr), transform)
        val_data_set = datasets.ImageFolder(os.path.join(data_path, te), transform)

        if args.model == 'swin':
            model = VitCifarmodel().to(device)
        elif args.model == 'vit':
            model = VitCifarmodel().to(device)
    print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
    return train_data_set, val_data_set, model


def main(args):
    open_wandb = args.open_wandb
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")


    # ????????????????????????
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # ????????????????????????GPU?????????????????????
    checkpoint_path = ""
    num_steps = args.num_steps
    fangfa = '4relu'
    conf = [args.dataset,fangfa,str(args.lr)]
    log_dir = '_'.join(conf)

    if open_wandb and rank == 0:
        # wandb.init(project="ann2snn", entity="alexandrewang")
        wandb.init(project="pretrain", entity="spikingtransformer", config=args)
        wandb.init(config=args)



    train_data_set, val_data_set, model = getdataset(args)

    if rank == 0:  # ????????????????????????????????????????????????tensorboard
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")
        # init_img = torch.zeros((1, 3, 32, 32), device=device)
        # tb_writer.add_graph(model, init_img)
        # summary(model, (1, 3, 32, 32))

    # ?????????rank??????????????????????????????????????????
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # ??????????????????batch_size?????????????????????list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=False)

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    nw = 4
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)

    # ????????????????????????????????????
    if os.path.exists(weights_path):
        # weights_dict = torch.load(weights_path, map_location=device)
        # load_weights_dict = {k: v for k, v in weights_dict.items()
        #                      if model.state_dict()[k].numel() == v.numel()}
        # model.load_state_dict(load_weights_dict, strict=False)

        # new_weights_dict = {}
        # weights_dict = torch.load(weights_path, map_location=device)
        # new_weights_dict = model.state_dict()
        # state_dict = {k:v for k,v in weights_dict.items() if k in new_weights_dict.keys()}
        # new_weights_dict.update(state_dict)
        # model.load_state_dict(new_weights_dict, strict=False)


        weights_dict = torch.load(weights_path, map_location=device)
        new_weights_dict = model.state_dict()
        for k, v in model.state_dict().items():
            if k in weights_dict.keys():
                if new_weights_dict[k].numel() == weights_dict[k].numel():
                    new_weights_dict[k] = weights_dict[k]
            else:
                print(k)
        model.load_state_dict(new_weights_dict, strict=False)


        # weights_dict = torch.load(args.weights, map_location=device)["model"]
        # # ?????????????????????????????????
        # for k in list(weights_dict.keys()):
        #     if "head" in k:
        #         del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))
        # print("load weights!")
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # ??????????????????????????????map_location?????????????????????????????????GPU??????????????????
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ??????????????????
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # ??????????????????????????????????????????????????????
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # ??????????????????BN????????????????????????SyncBatchNorm????????????
        if args.syncBN:
            # ??????SyncBatchNorm?????????????????????
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # ??????DDP??????
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # optimizer
    # pg = [p for p in model.parameters() if p.requires_grad]S
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=args.lr, weight_decay=0.05)
    scheduler = WarmupLR(optimizer=optimizer, num_warm=10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=5e-6)
    # scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=args.epochs,
    #         t_mul=1.,
    #         lr_min=5e-6,
    #         warmup_lr_init=5e-7,
    #         warmup_t=10,
    #         cycle_limit=1,
    #         t_in_epochs=False,
    #     )
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        mean_loss, train_sum_num = train_one_epoch(model=model,
                                                   optimizer=optimizer,
                                                   data_loader=train_loader,
                                                   device=device,
                                                   epoch=epoch,
                                                   num_steps=num_steps)
        train_acc = train_sum_num / train_sampler.total_size
        if open_wandb and rank==0:
            wandb.log({"train_loss": mean_loss})
            wandb.log({"train_acc": train_acc})
            wandb.log({'lr': float(optimizer.state_dict()['param_groups'][0]['lr'])})

        # Optional

        scheduler.step()

        loss, sum_num, = evaluate(model=model,
                                  data_loader=val_loader,
                                  device=device,
                                  num_steps=num_steps,
                                  epoch=epoch)
        acc = sum_num / val_sampler.total_size
        if open_wandb and rank==0:
            wandb.log({"test_loss": loss})
            wandb.log({"test_acc": acc})


        if rank == 0:
            print("[epoch {}] train_accuracy: {} test_accuracy: {}".format(epoch, round(train_acc, 4), round(acc, 4)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            if best_acc < acc:
                best_acc = acc
                # best_max_act = model.record()
                # thresh_list = np.load('best_max_threshold.npy')
                # np.save('best_max_threshold_best', thresh_list)
                # print(thresh_list)
                # torch.save(model.module.state_dict(), "./weights2/best_model.pth")
                torch.save(model.module.state_dict(), log_dir +'/epoch%d_%.3f.pth' % (epoch,
                                                                       acc))
                print('saving...')

    # ????????????????????????
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()
    if open_wandb:
        wandb.watch(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=90)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    # ????????????SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--open_wandb", type=bool, default=True)

    # ????????????????????????
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="/hpc/users/CONNECT/q18010301/imagenet")
    parser.add_argument('--dataset', type=str, default="imagenet")
    parser.add_argument('--model', type=str, default="swin")

    # resnet34 ????????????????????????   ../weights/vit_base_patch16_224.pth
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='/hpc/users/CONNECT/q18010301/ziqing/vision_transformer/train_multi_GPU/Cifar10_4reluV2_0.0001/epoch76_0.960.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # ??????????????????????????????????????????
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # ??????????????????(??????????????????),?????????????????????????????????nproc_per_node????????????
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
