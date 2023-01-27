# Byte Pair Encoding for Symbolic Music

Code of the paper *Byte Pair Encoding for Symbolic Music*.

## Steps to reproduce

1. `pip install -r requirements` to install requirements
2. Download the [GiantMIDI](https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md) dataset and put it in `data/`
3. `sh scripts/download_pop909.sh` to download and preprocess the [POP909](https://github.com/music-x-lab/POP909-Dataset) dataset
4. `python scripts/tokenize_datasets.py` to tokenize data and learn BPE
5. `python exp_gen.py` to train generative models and generate results
6. `python exp_cla.py` to train classification models and test them

[Scripts](./scripts) can be run to get reproduce the analysis.

## BPE learning

<img src="figures/tokenizations_bpe_token_types/POP909-merged_TSD.png" alt="POP909 TSD" width="400"/><img src="figures/tokenizations_bpe_token_types/POP909-merged_REMI.png" alt="POP909 REMI" width="400"/>

<img src="figures/tokenizations_bpe_token_types/GiantMIDI_TSD.png" alt="GiantMIDI TSD" width="400"/><img src="figures/tokenizations_bpe_token_types/GiantMIDI_REMI.png" alt="GiantMIDI REMI" width="400"/>

By orders, figures above are for POP909 TSD, POP909 REMI, GiantMIDI TSD, GiantMIDI REMI

<img src="figures/bpe_nb_tok_combinations.png" alt="GiantMIDI REMI" width="800"/>

## Experiment results

We refer you to the tables of the paper.

## Learned embedding space

### Singular values

#### Generators : POP909 TSD, POP909 REMI, GiantMIDI TSD and GiantMIDI REMI

<img src="figures/singular_value_gen/singular_value_POP909-merged_TSD.png" alt="POP909 TSD" width="200"/><img src="figures/singular_value_gen/singular_value_POP909-merged_REMI.png" alt="POP909 REMI" width="200"/><img src="figures/singular_value_gen/singular_value_GiantMIDI_TSD.png" alt="GiantMIDI TSD" width="200"/><img src="figures/singular_value_gen/singular_value_GiantMIDI_REMI.png" alt="GiantMIDI REMI" width="200"/>

#### Classifiers : $\mathrm{Cla}\_{small}$ TSD, $\mathrm{Cla}\_{small}$ REMI, $\mathrm{Cla}\_{large}$ TSD and $\mathrm{Cla}\_{large}$ REMI

<img src="figures/singular_value_cla/singular_value_GiantMIDI_TSD.png" alt="Cla small TSD" width="200"/><img src="figures/singular_value_cla/singular_value_GiantMIDI_REMI.png" alt="Cla small REMI" width="200"/><img src="figures/singular_value_cla/singular_value_GiantMIDI_TSD_LARGE.png" alt="Cla large TSD" width="200"/><img src="figures/singular_value_cla/singular_value_GiantMIDI_REMI_LARGE.png" alt="Cla large REMI" width="200"/>

### UMAP Generators

Figures are by order for no BPE, BPEx4, BPEx10, BPEx20, BPEx50, BPEx100, PVm and PVDm.

#### POP909 TSD

<img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_noBPE.png" alt="No BPE" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_bpe4.png" alt="BPEx4" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_bpe10.png" alt="BPEx10" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_bpe20.png" alt="BPEx20" width="200"/>

<img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_bpe50.png" alt="BPEx50" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_bpe100.png" alt="BPEx100" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_PVm.png" alt="PVm" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_TSD_PVDm.png" alt="PVDm" width="200"/>

#### POP909 REMI

<img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_noBPE.png" alt="No BPE" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_bpe4.png" alt="BPEx4" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_bpe10.png" alt="BPEx10" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_bpe20.png" alt="BPEx20" width="200"/>

<img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_bpe50.png" alt="BPEx50" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_bpe100.png" alt="BPEx100" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_PVm.png" alt="PVm" width="200"/><img src="figures/umap_3d_gen/umap_3d_POP909-merged_REMI_PVDm.png" alt="PVDm" width="200"/>

#### GiantMIDI TSD

<img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_noBPE.png" alt="No BPE" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_bpe4.png" alt="BPEx4" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_bpe10.png" alt="BPEx10" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_bpe20.png" alt="BPEx20" width="200"/>

<img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_bpe50.png" alt="BPEx50" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_bpe100.png" alt="BPEx100" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_PVm.png" alt="PVm" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_TSD_PVDm.png" alt="PVDm" width="200"/>

#### GiantMIDI REMI

<img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_noBPE.png" alt="No BPE" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_bpe4.png" alt="BPEx4" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_bpe10.png" alt="BPEx10" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_bpe20.png" alt="BPEx20" width="200"/>

<img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_bpe50.png" alt="BPEx50" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_bpe100.png" alt="BPEx100" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_PVm.png" alt="PVm" width="200"/><img src="figures/umap_3d_gen/umap_3d_GiantMIDI_REMI_PVDm.png" alt="PVDm" width="200"/>


### UMAP Classifiers

These figures are for $\mathrm{Cla}\_{small}$ and TSD. More figures can be found in [figures](./figures).

<img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_noBPE.png" alt="No BPE" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_bpe4.png" alt="BPEx4" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_bpe10.png" alt="BPEx10" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_bpe20.png" alt="BPEx20" width="200"/>

<img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_bpe50.png" alt="BPEx50" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_bpe100.png" alt="BPEx100" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_PVm.png" alt="PVm" width="200"/><img src="figures/umap_2d_cla/umap_2d_GiantMIDI_TSD_PVDm.png" alt="PVDm" width="200"/>

### Intrinsic dimension

#### Generators : POP909 TSD, POP909 REMI, GiantMIDI TSD and GiantMIDI REMI

<img src="figures/intrinsic_dimension_gen/intrinsic_dim_POP909-merged_TSD.png" alt="POP909 TSD" width="200"/><img src="figures/intrinsic_dimension_gen/intrinsic_dim_POP909-merged_REMI.png" alt="POP909 REMI" width="200"/><img src="figures/intrinsic_dimension_gen/intrinsic_dim_GiantMIDI_TSD.png" alt="GiantMIDI TSD" width="200"/><img src="figures/intrinsic_dimension_gen/intrinsic_dim_GiantMIDI_REMI.png" alt="GiantMIDI REMI" width="200"/>

#### Classifiers : $\mathrm{Cla}\_{small}$ TSD, $\mathrm{Cla}\_{small}$ REMI, $\mathrm{Cla}\_{large}$ TSD and $\mathrm{Cla}\_{large}$ REMI

<img src="figures/intrinsic_dimension_cla/intrinsic_dim_GiantMIDI_TSD.png" alt="Cla small TSD" width="200"/><img src="figures/intrinsic_dimension_cla/intrinsic_dim_GiantMIDI_REMI.png" alt="Cla small REMI" width="200"/><img src="figures/intrinsic_dimension_cla/intrinsic_dim_GiantMIDI_TSD_LARGE.png" alt="Cla large TSD" width="200"/><img src="figures/intrinsic_dimension_cla/intrinsic_dim_GiantMIDI_REMI_LARGE.png" alt="Cla large REMI" width="200"/>
