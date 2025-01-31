import math
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl

from models.probabilistic_unet.probabilistic_unet import ProbabilisticUnet
from models.probabilistic_unet.utils import l2_regularisation
from datasets.optical_flow_dataset import OpticalFlowPair



class UnetVAETrainer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.prob_unet = ProbabilisticUnet(input_channels=2, num_classes=2, num_filters=[32, 64, 128, 192], latent_dim=self.hparams.latent_dim, no_convs_fcomb=4, beta=10.0)
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        flow_prev, body_flow_prev, flow, body_flow, annotation = batch
        self.prob_unet(flow_prev, flow, training=self.training)

        if self.training:
            elbo = self.prob_unet.elbo(flow)
            recon = self.prob_unet.reconstruction
            reg_loss = l2_regularisation(self.prob_unet.posterior) + l2_regularisation(self.prob_unet.prior) + l2_regularisation(self.prob_unet.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
        else:
            recon = self.prob_unet.sample(testing=not self.training)
            loss = None

        return recon, loss

    def training_step(self, batch, batch_idx):
        recon, loss = self(batch)
        self.train_outputs.append(loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        flow_prev, body_flow_prev, flow, body_flow, annotation = batch
        recon, _ = self(batch)
        loss = F.mse_loss(recon, flow)
        log = {'loss': loss}
        self.val_outputs.append(log)
        return log

    def test_step(self, batch, batch_idx):
        flow_prev, body_flow_prev, flow, body_flow, annotation = batch
        recon, _ = self(batch)
        loss = F.mse_loss(recon, flow)
        log = {'loss': loss}
        self.test_outputs.append(log)
        return log

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_outputs = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
        self.log("val_loss", avg_loss)
        self.val_outputs.clear()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_outputs = []

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log("test_loss", avg_loss)
        self.test_outputs.clear()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_outputs = []

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_outputs).mean()
        self.log("train_loss", avg_loss)
        self.train_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.hparams.sample_size),
                transforms.CenterCrop(self.hparams.sample_size),
                transforms.ToTensor(),
            ]
        )

        train_dataset = OpticalFlowPair(self.hparams.video_root, flow_type=self.hparams.flow_type, frame_offset_start=self.hparams.prediction_offset_start, frame_offset=self.hparams.prediction_offset, transform=transform)
        if self.training:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=True, num_workers=self.hparams.n_threads, pin_memory=True)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                               shuffle=False, num_workers=self.hparams.n_threads, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.hparams.sample_size),
                transforms.CenterCrop(self.hparams.sample_size),
                transforms.ToTensor(),
            ]
        )

        val_dataset = OpticalFlowPair(self.hparams.val_video_root, flow_type=self.hparams.flow_type, frame_offset_start=self.hparams.prediction_offset_start, frame_offset=self.hparams.prediction_offset, transform=transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=True)
    def test_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.hparams.sample_size),
                transforms.CenterCrop(self.hparams.sample_size),
                transforms.ToTensor(),
            ]
        )
        test_dataset = OpticalFlowPair(self.hparams.test_video_root, flow_type=self.hparams.flow_type, frame_offset_start=self.hparams.prediction_offset_start, frame_offset=self.hparams.prediction_offset, transform=transform, train=False)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.n_threads, pin_memory=True)
