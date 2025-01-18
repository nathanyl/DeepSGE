from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from GATv2 import GATv2Conv
from transformer import ViT
from utils import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet34(pretrained=True).to(device)
class DeepSGE(pl.LightningModule):
    def __init__(self, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.2, pos_dim=64,
                 heads=[16, 8], out_dim=64, use_DropKey=True, mask_ratio=0.4):
        super().__init__()

        self.learning_rate = learning_rate
        self.out_dim = out_dim
        self.relu = nn.ReLU()

        self.x_embed = nn.Embedding(pos_dim, out_dim)
        self.y_embed = nn.Embedding(pos_dim, out_dim)

        self.vit = ViT(dim=1128, depth=n_layers, heads=heads[0], mlp_dim=2 * dim, dropout=dropout,
                       emb_dropout=dropout, use_DropKey=use_DropKey, mask_ratio=mask_ratio)

        self.gatv2 = GATv2Conv(in_channels=1128, out_channels=64, heads=8, dropout=dropout,)

        self.gene_out = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_genes)
        )

    def forward(self, patches, centers, adj):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        adj = torch.squeeze(adj, dim=0)

        patches = resnet(patches)
        patches = self.relu(patches)

        centers_x = self.x_embed(centers[:, :, 0]).permute(1, 0, 2)#n*1*64
        centers_y = self.y_embed(centers[:, :, 1]).permute(1, 0, 2)
        centers_x = torch.squeeze(centers_x, dim=1)
        centers_y = torch.squeeze(centers_y, dim=1)

        x = torch.concat((patches, centers_x, centers_y), dim=1)
        x = x.reshape(1, x.shape[0], -1)
        x = self.vit(x)

        x = x.reshape(x.shape[1], -1)
        x = self.gatv2(x, adj)
        x = self.gene_out(x)
        return x

    def aug(self, patch, center, adj):
        patch = patch.squeeze(0)
        aug_patch = torch.zeros_like(patch)
        for i in range(aug_patch.shape[0]):
            if random.random() < 1:
                aug_patch[i] = rotate_image(patch[i])
        aug_patch = aug_patch.unsqueeze(0)
        x = self(aug_patch, center, adj)
        return x

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp, adj = batch
        pred = self(patch, center, adj)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        patch, position, exp, center, adj = batch
        pred = self(patch, position, adj)
        preds = pred.squeeze()
        ct = center
        gt = exp
        preds = preds.cpu().squeeze().numpy()
        ct = ct.cpu().squeeze().numpy()
        gt = gt.cpu().squeeze().numpy()
        adata = ann.AnnData(preds)
        adata.obsm['spatial'] = ct

        adata_gt = ann.AnnData(gt)
        adata_gt.obsm['spatial'] = ct

        return adata, adata_gt

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
