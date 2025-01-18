# DeepSGE
DeepSGE
# Requirements
- python = 3.9.19
- pytorch = 2.4.1
- pytorch-geometric = 2.5.3
- sklearn = 1.3.0
# Installation
Download DeepSGE by
```
git clone https://github.com/nathanyl/DeepSGE
```
## Run DeepSGE model
```
python train.py 
```
```
 parameters:  
 - `n_genes`: int.  
  Amount of genes.
 - `heads`: int, default `[16, 8]`.  
  The number of heads of the Vit module and the number of heads of the GATv2 module.
 - `n_layers`: int, default `4`.  
  Number of Transformer blocks.
 arguments:
 - `test_sample_ID`: the test sample id of the validation set for cross-validation
 - `vit_dataset`: how the dataset is loaded
 - `modelsave_address`: the storage address of the training model
 - `dataset_name`: the name of current using dataset      
```
## Datasets
The mouse olfactory bulb dataset is available for download at:
https://www.spatialresearch.org/resources-published-datasets/doi-10-1126science-aaf2403/

The human cutaneous squamous cell carcinoma dataset can be accessed
 through the GEO database under accession number GSE144240.

The human HER2-positive breast cancer datasets are available
 for download at: https://github.com/almaan/her2st/

The human dorsolateral prefrontal cortex dataset can be found
 at: http://research.libd.org/spatialLIBD/
 
