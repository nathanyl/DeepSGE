import csv
import os
import random

import anndata as ad
import anndata as ann
import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
from PIL import Image
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import mean_squared_error
Image.MAX_IMAGE_PIXELS = 933120000

BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']
IG = {'B_cell': BCELL, 'Tumor': TUMOR, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T, 'Dendritic_cells': DC,
      'Mature_dendritic_cells': MDC, 'Cutaneous_Malignant_Melanoma': CMM}
MARKERS = []
for i in IG.values():
    MARKERS += i
LYM = {'B_cell': BCELL, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T}

def rotate_image(image_tensor):
    # 随机选择旋转角度
    angle = 90 if random.random() < 0.5 else 180
    # 随机旋转图像
    if angle == 90:
        return torch.rot90(image_tensor, k=1, dims=(1, 2))  # 90°逆时针旋转
    else:
        return torch.rot90(image_tensor, k=2, dims=(1, 2))  # 180°逆时针旋转


def get_R(data1, data2, dim=1, func=mean_squared_error):
    adata1 = data1.X
    adata2 = data2.X
    mse_values = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            mse = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            mse = func(adata1[g, :], adata2[g, :])
        mse_values.append(mse)
    mse_values = np.array(mse_values)

    def write_to_csv(filename, data):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow([row])

    filename = 'mse_values.csv'
    write_to_csv(filename, mse_values)
    return mse_values, mse_values

def cluster(adata, label):
    idx = label != 'undetermined'
    tmp = adata[idx]
    l = label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0, n_init=20).fit(tmp.obsm['X_pca'])
    p = kmeans.labels_.astype(str)
    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    adata.obs['kmeans'] = lbl
    return p, round(ari_score(l, p), 3)


