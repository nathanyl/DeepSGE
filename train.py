# coding:utf-8 
import random
import scanpy as sc
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dataset import ViT_HER2ST, ViT_SKIN, ViT_DLPC, ViT_MOB
from predict import model_predict
from utils import *

from model import DeepSGE


def train(test_sample_ID, vit_dataset, epochs, modelsave_address, dataset_name):
    dataset = vit_dataset(train=True, fold=test_sample_ID)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    if dataset_name == "her2st":
        mid_name = "-her2st_785_32_cv"
        model = DeepSGE(n_genes=785, learning_rate=1e-5, out_dim=64, heads=[16, 8], n_layers=4)
    elif dataset_name == "cscc":
        mid_name = "-cscc_171_12_cv"
        model = DeepSGE(n_genes=171, learning_rate=1e-5, out_dim=64, heads=[16, 8], n_layers=8)
    elif dataset_name == "dlpc":
        mid_name = "-dlpc_161_12_cv"
        model = DeepSGE(n_genes=161, learning_rate=1e-5, pos_dim=128, out_dim=64, heads=[16, 8], n_layers=8)
    else:
        mid_name = "-mob_119_12_cv"
        model = DeepSGE(n_genes=119, learning_rate=1e-5, out_dim=64, heads=[16, 8], n_layers=8)
    mylogger = CSVLogger(save_dir=modelsave_address + "/../logs/",
                         name="my_test_log_" + mid_name + '_' + str(test_sample_ID))
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=epochs,
                         logger=mylogger)

    trainer.fit(model, train_loader)

    dataset_test = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=0)

    pred, gt = trainer.predict(model=model, dataloaders=test_loader)[0]
    R, p_val = get_R(pred, gt)
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    print('Mean Pearson Correlation:', np.nanmean(R))
    trainer.save_checkpoint(modelsave_address+"/"+"DeepSGE"+mid_name+'_'+str(test_sample_ID)+".ckpt")


def test(test_sample_ID, vit_dataset, model_address, dataset_name):
    if dataset_name == "her2st":
        mid_name = "-her2st_785_32_cv"
        g = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        model = DeepSGE.load_from_checkpoint(
            model_address + "/DeepSGE" + mid_name + '_' + str(test_sample_ID) + ".ckpt", n_genes=785,
            learning_rate=1e-5, route_dim=64, heads=[16, 8],
            n_layers=4)
    elif dataset_name == "cscc":
        mid_name = "-cscc_171_12_cv"
        g = list(np.load('./data/skin_hvg_cut_1000.npy', allow_pickle=True))
        model = DeepSGE.load_from_checkpoint(
            model_address + "/DeepSGE" + mid_name + '_' + str(test_sample_ID) + ".ckpt", n_genes=171,
            learning_rate=1e-5, route_dim=64, heads=[16, 8],
            n_layers=8)
    elif dataset_name == "dlpc":
        mid_name = "-dlpc_161_12_cv"
        g = list(np.load('./data/dlpc_hvg_cut_1000.npy', allow_pickle=True))
        model = DeepSGE.load_from_checkpoint(
            model_address + "/DeepSGE" + mid_name + '_' + str(test_sample_ID) + ".ckpt", n_genes=161,
            learning_rate=1e-5, pos_dim=128, route_dim=64, heads=[16, 8],
            n_layers=8)
    else:
        mid_name = "-mob_119_12_cv"
        g = list(np.load('data/mob_hvg_cut_1000.npy', allow_pickle=True))
        model = DeepSGE.load_from_checkpoint(
            model_address + "/DeepSGE" + mid_name + '_' + str(test_sample_ID) + ".ckpt", n_genes=119,
            learning_rate=1e-5, route_dim=64, heads=[16, 8],
            n_layers=8)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = vit_dataset(train=False, sr=False, fold=test_sample_ID)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    adata_pred, adata_truth = model_predict(model, test_loader, device=device)

    adata_pred.var_names = g
    adata_truth.var_names = g
    sc.pp.scale(adata_pred)
    if test_sample_ID in [5, 11, 17, 23, 26,
                          30] and dataset_name == 'her2st':
        label = dataset.label[dataset.names[0]]
        return adata_pred, adata_truth, label
    else:
        return adata_pred, adata_truth


if __name__ == '__main__':
    seed = 10000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for i in range(0, 1):
    #     train(i, ViT_HER2ST, 300, "model", "her2st")
    #
    # for i in range(0, 1):
    #     train(i, ViT_SKIN, 300, "model", "cscc")

    # for i in range(0, 12):
    #     train(i, ViT_DLPC, 300, "model", "dlpc")
    #
    for i in range(0, 1):
        train(i, ViT_MOB, 300, "model", "mob")

    for i in range(0, 1):
        dataset = 'cscc'
        test_sample = i
        if dataset == "her2st":
            if test_sample in [5, 11, 17, 23, 26, 30]:
                pred, gt, label = test(test_sample, ViT_HER2ST, "model", dataset)
            else:
                pred, gt = test(test_sample, ViT_HER2ST, "model", dataset)
            R, p_val = get_R(pred, gt)

            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            pred.var["pcc"] = R
            print('Mean Pearson Correlation:', np.nanmean(R))


        elif dataset == "cscc":
            pred, gt = test(test_sample, ViT_SKIN, "model", dataset)
            R, p_val = get_R(pred, gt)
            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            pred.var["pcc"] = R
            print('Mean Pearson Correlation:', np.nanmean(R))
            # gene spatial representation
            im = np.array(Image.open('data/GSE144240_RAW/GSM4284323_P9_ST_rep2.jpg'))
            sc.pl.spatial(pred, img=im, color='PI3', spot_size=112, color_map='magma', bw=True)
            # sc.pl.spatial(gt, img=im, color='PI3', spot_size=112, color_map='magma', bw=True)
        elif dataset == "dlpc":
            pred, gt = test(test_sample, ViT_DLPC, "model", dataset)
            R, p_val = get_R(pred, gt)
            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            pred.var["pcc"] = R
            print('Mean Pearson Correlation:', np.nanmean(R))

        else:
            pred, gt = test(test_sample, ViT_MOB, "model", dataset)
            R, p_val = get_R(pred, gt)
            pred.var["p_val"] = p_val
            pred.var["-log10p_val"] = -np.log10(p_val)
            pred.var["pcc"] = R
            print('Mean Pearson Correlation:', np.nanmean(R))
