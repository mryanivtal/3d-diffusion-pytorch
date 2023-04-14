# Novel View Synthesis with Diffusion Models

Unofficial PyTorch Implementation of [Novel View Synthesis with Diffusion Models](https://3d-diffusion.github.io/).

## Changes:

As the JAX code given by the authors are not runnable, we fixed the original code to runnable JAX code, while following the authors intend described in the paper. Please compare [Author's code](original_jax_code/original_code.py.bak) and our [fixed version](original_jax_code/original_code_fixed.py.bak) to see the changes we made.

The PyTorch implementation is in [xunet.py](model/xunet.py). Feel free to put up an issue if the implementation is not consistent with the original JAX code. 

## Data Preparation:

Visit [SRN repository](https://github.com/vsitzmann/scene-representation-networks) and download `chairs_train.zip` and `cars_train.zip` and extract the downloaded files in `/data/`. Here we use 90% of the training data for training and 10% as the validation set.

We include pickle file that contains available view-png files per object. 

## Training:
To get help:
```
python train.py --help
```

To train from scratch:
```
python train.py --datadir="<your-data-dir>" --outdir="<your-output-dir>" --batchsize=32 --reportlossevery=32 --evaluateevery=20 --checkpointevery=10 --epochs=50
```

To continue training from checkpoint:

```
python train.py --datadir="<your-data-dir>" --checkpointdir="<path-of-stored-checkpoint>" --batchsize=32 --reportlossevery=32 --evaluateevery=20 --checkpointevery=10 --epochs=50
```

## Sampling:

```
python sample.py --model="<path-to-model>" --refimagedir="<your-data-dir>/a4d535e1b1d3c153ff23af07d9064736" --outdir=",output-path-for-samples>"
```

We set the diffusion steps to be 256 during sampling procedure, which takes around 10 minutes per view. 

## Missing:
1. EMA decay not implemented.
2. This is still a work in progress, generally works
