# Code for "Learning Deep Disentangled Embeddings with the F-Statistic Loss" (NIPS 2018)

[Read the paper](https://arxiv.org/abs/1802.05312)

## Pre-Reqs
1. Tensorflow (tested on `v1.10.0`)
2. `pip install -r requirements.txt`

## Downloading datasets

1. [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
2. [Market-1501](http://www.liangzheng.org/Project/project_reid.html)
3. Small NORB ( parts [1](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz), [2](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz), [3](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz), [4](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz), [5](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz), [6](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz) )
4. [Sprites](http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz)
5. CUB-200 with Inception V3 Features (in root folder, run `sh data/download_cub.sh`)

## Extracting datasets
These scripts each extract one `npz` type file into the `data/` subfolder.
1. CUHK03: `python data/cuhk03.py <path-to-cuhk03_release>/`
2. Market-1501: `python data/market1501.py <path-to-Market-1501-v15.09.15>/`
3. Small NORB: `python data/small_norb.py <path-to-small_norb>/`
4. Sprites: `python data/sprites.py <path-to-nips2015_analogy>/sprites/`
5. CUB-200 needs no further extraction

## Training embeddings
The configurations used in the paper are available in the `example_configs` subfolder. For example, to train a sprite model with a factor-aware oracle and F-Statistic loss embedding:

`python embedding.py @example_configs/f/sprites_factor/opts.txt -c train`

Configurations from the paper:
```
cub200/<f,triplet,histogram,lsss,binomial_dev>
cuhk03/<f,triplet,histogram,lsss,binomial_dev>
sprites/class/f **class-aware training for sprites**
sprites/factor/<f,triplet,histogram> **factor-aware training for sprites**
market1501/<f,triplet,histogram,lsss,binomial_dev>
small_norb/<f,triplet,histogram>
```

Each of these examples only trains the first split. To train a split `j`, change the `-xvs j` parameter.

## Evaluating Recall@K
Run `python embedding.py @path/to/model/opts.txt -me -1 -c test`. 

Recall@K results for train/val/test splits will be saved to `path/to/model/test.txt`.

## Evaluating modularity and explicitness
_This only works for the data sets with factor labels (Sprites and Small-NORB)_

Run the following two scripts:
1. Modularity: `python eval/modularity.py @path/to/model/opts.txt -me -1 -c test`
** Modularity scores (for each dimension) will be saved to `path/to/model/modularity_test.txt`
2. Explicitness: `python eval/explicitness.py @path/to/model/opts.txt -me -1 -c test`
** Explicitness scores (for each factor) will be saved to `path/to/model/explicitness_test.txt`
