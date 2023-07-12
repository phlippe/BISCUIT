# 🍪 BISCUIT: Causal Representation Learning from Binary Interactions

[![ProjectPage](https://img.shields.io/static/v1.svg?logo=html&label=Website&message=Project%20Page&color=red)](https://phlippe.github.io/BISCUIT/)
[![Demo](https://colab.research.google.com/assets/colab-badge.svg?label=Demo)](https://colab.research.google.com/github/phlippe/BISCUIT/blob/main/demo.ipynb)
[![Paper](https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green)](https://arxiv.org/abs/2306.09643)
[![Datasets](https://img.shields.io/static/v1.svg?logo=zenodo&label=Zenodo&message=Download%20Datasets&color=blue)](https://zenodo.org/record/8027138)   

This is the official code repository for the papers **"BISCUIT: Causal Representation Learning from Binary Interactions"** (UAI 2023), by Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M. Asano, Taco Cohen, and Efstratios Gavves. For a paper summary, check out our [project page](https://phlippe.github.io/BISCUIT/)!

## Requirements

The code was originally written for PyTorch v1.13 and Python 3.8. Higher versions of PyTorch and Python are expected to work as well (tested on v2.0.1).
Further, the code base makes use of PyTorch Lightning (tested for v1.9 for PyTorch v1.13 and v2.0.3 for PyTorch v2.0.1) to structure the training code.
The steps for installing the requirements are:

1. Create a new conda environment from the provided YAML file:
   ```setup
   conda env create -f environment.yml
   ```
   In case conda raises issues during the installation of the packages, please install the packages individually. The main packages are PyTorch, PyTorch Lightning, AI2Thor, CausalWorld, and matplotlib.
   
2. Activate the environment
   ```setup
   conda activate biscuit
   ```

## Demo

We provide a demo notebook that showcases a pretrained BISCUIT-NF model on the iTHOR dataset. The notebook can be run on Google Colab and can be found [here](https://colab.research.google.com/github/phlippe/BISCUIT/blob/main/demo.ipynb).

![Triplet Generation](https://github.com/phlippe/BISCUIT/assets/25037725/882e5258-74b4-4790-aa75-341e7059230e)


## Datasets

We provide the data generation scripts for reproducing our papers in the folder `data_generation/`. See below for the specific commands to reproduce the specific datasets used in the paper. Alternatively, the datasets can be downloaded from our zenodo upload (see badge above).

### iTHOR Embodied AI

To generate the iTHOR dataset, simply run:
```
python data_generation/data_generation_ithor.py
```
For training, set `num_sequences` to 1500 and `prefix` to `"train"`, and for evaluation, set `num_sequences` to 250 and `prefix` to `"val"`/`"test"`.

In case you face difficulties with installing or running the simulation via the `ai2thor` package, please refer to the [official documentation](https://ai2thor.allenai.org/ithor/documentation/). 

https://github.com/phlippe/BISCUIT/assets/25037725/c0826868-2a93-4c92-a504-03627efe884c

### CausalWorld

To generate the CausalWorld dataset, run:
```
python data_generation/data_generation_causal_world.py --output_folder data/causal_world \
                                                       --num_sequences 200 \
                                                       --seq_len 1000 \
                                                       --num_triplet_sets 10 \
                                                       --num_indep_sets 25 \
                                                       --num_processes 30
```
Adjust the parameter `num_processes` to the number of cores available on your machine. 

The CausalWorld dataset generation may require a slightly adjusted version of the `causal_world` package. If interested in obtaining this code, please reach out (see contact information at the bottom). 

https://github.com/phlippe/BISCUIT/assets/25037725/3a585c18-cb48-4f17-bee1-1f5dd6dd1f34

### Voronoi

To generate the Voronoi dataset with 6 variables and standard interactions (i.e. robotic arm on Voronoi tiles), run:
```
python data_generation_voronoi.py --output_folder ../data/voronoi/6vars_random_interactions_seed42 \
                                  --dataset_size 150000 \
                                  --num_causal_vars 6 \
                                  --edge_prob_instant 0.0 \
                                  --edge_prob_temporal 0.4 \
                                  --num_flow_layers 2 \
                                  --use_action_based_interventions \
                                  --num_processes 24
```
Adjust the parameter `num_processes` to the number of cores available on your machine.
For 9 variables, change the parameter `num_causal_vars` to 9. 
To generate a dataset with minimal number of interactions, add `--use_minimal_action_interventions`.

![Voronoi_Example](https://github.com/phlippe/BISCUIT/assets/25037725/570a47bd-a564-419a-a4f1-56c576738c27)

## Running experiments

The repository is structured in three main folders:

* `experiments` contains all utilities for running experiments.
* `models` contains the code of BISCUIT.
* `data_generation` contains all utilities for creating the dataset.

For running an experiments, we below for the training commands.

### Autoencoder

For the CausalWorld and iTHOR dataset, we provide a pretrained autoencoder with the published datasets. Alternatively, you can train an autoencoder yourself with `experiments/train_ae.py`. Once the autoencoder is trained, an NF can be trained to map it to a causal representation. For CausalWorld, use:
```
python train_ae.py --data_dir DATA_DIR \
                   --batch_size 128 \
                   --c_hid 128 \
                   --lr 5e-4 \
                   --warmup 100 \
                   --num_latents 32 \
                   --cluster \
                   --regularizer_weight 1e-5 \
                   --max_epochs 250 \
                   --seed 42
```
For iTHOR, use:
```
python train_ae.py --data_dir DATA_DIR \
                   --batch_size 64 \
                   --c_hid 64 \
                   --lr 2e-4 \
                   --warmup 1000 \
                   --num_latents 40 \
                   --cluster \
                   --regularizer_weight 4e-6 \
                   --max_epochs 100 \
                   --seed 42

```
Note that the datasets are identified via its name, i.e. iTHOR datasets should have 'ithor' in their folder name, and CausalWorld should have 'causal_world'.

### BISCUIT-NF

Once the autoencoder has been trained on CausalWorld or iTHOR, BISCUIT-NF can be trained on the respective dataset. For CausalWorld, use:
```
python train_nf.py --data_dir DATA_DIR \
                   --autoencoder_checkpoint CHECKPOINT_OF_AE \
                   --num_latents 32 \
                   --c_hid 64 \
                   --num_flows 6 \
                   --lr 1e-3 \
                   --prior_action_add_prev_state \
                   --num_samples 2 \
                   --batch_size 1024 \
                   --warmup 100 \
                   --seed 42
```
For iTHOR, the same command can be used, up to the `--num_latents` argument, which should be set to 40.

### BISCUIT-VAE

For training BISCUIT-VAE on the Voronoi datasets with 6 variables, use:
```
python train_vae.py --data_dir DATA_DIR \
                    --c_hid 32 \
                    --batch_size 256 \
                    --lr 1e-3 \
                    --num_latents 12 \
                    --max_epochs 100 \
                    --check_val_every_n_epoch 20 \
                    --cluster
```
For datasets with 9 variables, the same command can be used, up to the `--num_latents` argument, which should be set to 18.

## Citation

If you use this code or find it otherwise helpful, please consider citing our work:
```bibtex
@inproceedings{lippe2023biscuit,
   title        = {{BISCUIT: Causal Representation Learning from Binary Interactions}},
   author       = {Lippe, Phillip and Magliacane, Sara and L{\"o}we, Sindy and Asano, Yuki M and Cohen, Taco and Gavves, Efstratios},
   year         = 2023,
   booktitle    = {The 39th Conference on Uncertainty in Artificial Intelligence},
   url          = {https://openreview.net/forum?id=VS7Dn31xuB},
   abstract     = {Identifying the causal variables of an environment and how to intervene on them is of core value in applications such as robotics and embodied AI. While an agent can commonly interact with the environment and may implicitly perturb the behavior of some of these causal variables, often the targets it affects remain unknown. In this paper, we show that causal variables can still be identified for many common setups, e.g., additive Gaussian noise models, if the agent's interactions with a causal variable can be described by an unknown binary variable. This happens when each causal variable has two different mechanisms, e.g., an observational and an interventional one. Using this identifiability result, we propose BISCUIT, a method for simultaneously learning causal variables and their corresponding binary interaction variables. On three robotic-inspired datasets, BISCUIT accurately identifies causal variables and can even be scaled to complex, realistic environments for embodied AI.}
}
```

### Contact

If you have questions or found a bug, feel free to open a github issue or send a mail to p.lippe@uva.nl. 
