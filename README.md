# TransferCausalLearning
code for **Transfer Causal Learning: Causal Effect Estimation with Knowledge Transfer**


## environment setup
```bash
conda env create -f environment.yaml
```
## training model 

To reproduce paper experiments, the defualt parameters are set to reproduce the experiments reported in the paper.

```bash
cd TCL 
coda activate TCL
```
TO-CL 
```bash
python -m run.run_base
```

WS-TCL 
```bash
python -m run.run_warmstart
```

our L1-TCL
```bash
python -m run.run_transfer
```

To run experiments with other hyperparameters. 

```bash
python -m run.run_transfer --batchsize 32 --transfer_lr 0.001 --knob tarnet --l1reg 0.01
```

Our Pytorch version of Dragonnet and TARNet (the 3-headed variant adopted in original Dragonnet implementation) implememntation is built upon an [existing repository](https://github.com/alecmn/dragonnet-reproduced/tree/main) which extends the original TensorFlow version [Shi et al. (2019)](https://github.com/claudiashi57/dragonnet), and our Pytorch version is able to reproduce the results in the original TensorFlow version with the same hyperparameters.
## Evaluation

To calculate aggregated metrics reported in the paper

```bash
python print_results.py
```
