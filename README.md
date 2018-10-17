# Code for "Learning Deep Disentangled Embeddings with the F-Statistic Loss" (NIPS 2018)

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
The configurations used in the paper are available in the `example_configs` subfolder. For example, to train a sprite model with an unnamed-factor oracle:
`python embedding.py @example_configs/sprites_uf/opts.txt -c train`
Configurations from the paper:
* `cuhk03` - class-aware training with CUHK03
* `market1501` - class-aware training with Market-1501
* `cub200` - class-aware training with CUB-200-2011
* `sprites_class` - class-aware training with sprites
* `sprites_factor` - factor-aware training with sprites
* `norb_factor` - factor-aware training with small NORB

Each of these examples only trains the first split. To train a split `j`, change the `-xvs j` parameter.

## Evaluating modularity and explicitness
Run the following two scripts:
`python eval/modularity.py @path/to/model/opts.txt -me -1 -c test`
`python eval/explitness.py @path/to/model/opts.txt -me -1 -c test`
The results are stored in `path/to/model/modularity.txt` and `path/to/model/explitiness.txt`
