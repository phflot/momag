# RAFT wrappers based on the official RAFT github https://github.com/princeton-vl/RAFT

from os import mkdir
from os.path import join
from os.path import isdir, dirname
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from neurovc.raft.raft import RAFT
import neurovc.raft.evaluate as evaluate
import neurovc.raft.datasets as datasets
from neurovc.raft.utils.flow_viz import flow_to_image
from neurovc.raft.utils.utils import InputPadder

from inspect import getsourcefile
from os.path import abspath, exists

import numpy as np
import cv2

from urllib.request import urlretrieve

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


class RAFTOpticalFlow:
    def __init__(self, iters=20,
                 model=None,
                 path=None,
                 small=False,
                 mixed_precision=False,
                 alternate_corr=False):

        if model is None:
            model_path = join(dirname(abspath(getsourcefile(lambda: 0))), "models")
            model = join(model_path, "raft-casme2.pth")
        if not exists(model):
            if not isdir(model_path):
                mkdir(model_path)
            print("Model could not be found, downloading raft-casme2 model...")
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urlretrieve("https://cloud.hiz-saarland.de/s/McMNXZ5o7xteE6n/download/raft-casme2.pth", model)
            print("done.")

        args = argparse.Namespace(
            model=model,
            path=path,
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr
        )

        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        self.model = model.module
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.last_flow = None
        self.padder = None
        self.iters = iters

    def calc(self, ref, frame, flow=None):
        ref_torch = torch.from_numpy(ref).permute(2, 0, 1).float()
        ref_torch = ref_torch[None].to(self.device)
        frame_torch = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame_torch = frame_torch[None].to(self.device)

        if self.padder is None:
            self.padder = InputPadder(ref_torch.shape)
        ref_torch, frame_torch = self.padder.pad(ref_torch, frame_torch)

        flow_low, flow_up = self.model(ref_torch, frame_torch, iters=self.iters, flow_init=flow, test_mode=True)
        flow_up = self.padder.unpad(flow_up)
        self.last_flow = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()

        return self.last_flow

    def visualize(self, title="flow", flow=None):
        if flow is None and self.last_flow is not None:
            cv2.imshow(title, flow_to_image(self.last_flow))
        if flow is not None:
            cv2.imshow(title, flow_to_image(flow))


class RAFTLogger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


class RAFTTrainer:
    MAX_FLOW = 400
    VAL_FREQ = 5000

    def __init__(self, name='raft', stage=None, validation=None, restore_ckpt=None, gpus=[0], num_steps=120000,
                 batch_size=5, lr=0.0001, image_size=[384, 512], wdecay=.00005, gamma=0.8, mixed_precision=False,
                 iters=12, epsilon=1e-8, clip=1.0, dropout=0.0, add_noise=False, path='.', small=False):

        args = argparse.Namespace(
            name=name,
            stage=stage,
            validation=validation,
            restore_ckpt=restore_ckpt,
            gpus=gpus,
            num_steps=num_steps,
            batch_size=batch_size,
            lr=lr,
            image_size=image_size,
            wdecay=wdecay,
            gamma=gamma,
            mixed_precision=mixed_precision,
            iters=iters,
            epsilon=epsilon,
            clip=clip,
            dropout=dropout,
            add_noise=add_noise,
            path=path,
            small=small
        )
        self.args = args
        self.model = None

    @staticmethod
    def _sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics

    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _fetch_optimizer(self):
        """ Create the optimizer and learning rate scheduler """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr,
                                weight_decay=self.args.wdecay, eps=self.args.epsilon)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, self.args.lr, self.args.num_steps + 100,
                                                  pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

        return optimizer, scheduler

    def train(self, seed=1234):
        torch.manual_seed(seed)
        np.random.seed(seed)

        args = self.args

        if not isdir(join(args.path, 'checkpoints')):
            mkdir(join(args.path, 'checkpoints'))

        self.model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
        print("Parameter Count: %d" % self._count_parameters())

        if args.restore_ckpt is not None:
            self.model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

        self.model.cuda()
        self.model.train()

        if args.stage != 'chairs':
            self.model.module.freeze_bn()

        train_loader = datasets.fetch_dataloader(args)
        optimizer, scheduler = self._fetch_optimizer()

        total_steps = 0
        scaler = GradScaler(enabled=args.mixed_precision)
        logger = RAFTLogger(self.model, scheduler)

        VAL_FREQ = 5000
        add_noise = True

        should_keep_training = True
        while should_keep_training:

            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad()
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = self.model(image1, image2, iters=args.iters)

                loss, metrics = self._sequence_loss(flow_predictions, flow, valid, args.gamma)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                logger.push(metrics)

                if total_steps % VAL_FREQ == VAL_FREQ - 1:
                    PATH = join(args.path, 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                    torch.save(self.model.state_dict(), PATH)

                    results = {}
                    for val_dataset in args.validation:
                        if val_dataset == 'chairs':
                            results.update(evaluate.validate_chairs(self.model.module))
                        elif val_dataset == 'sintel':
                            results.update(evaluate.validate_sintel(self.model.module))
                        elif val_dataset == 'kitti':
                            results.update(evaluate.validate_kitti(self.model.module))

                    logger.write_dict(results)

                    self.model.train()
                    if args.stage != 'chairs':
                        self.model.module.freeze_bn()

                total_steps += 1

                if total_steps > args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        PATH = join(args.path, 'checkpoints/%s.pth' % args.name)
        torch.save(self.model.state_dict(), PATH)

        return PATH
