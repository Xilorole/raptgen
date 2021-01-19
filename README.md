# Supplementary code for "RaptGen: A variational autoencoder with profile hidden Markov model for generative aptamer discovery"

## Installation
Run 

```shell
$ pip install -r requirements.txt
```

You also need to install `cairo` library to generate profile hmm image. For mac OS X, it can be installed by `brew install cairo && brew install pango`. For Ubuntu `sudo apt-get install -y libcairo2` would work.



## Quickstart

### Evaluating multi-motifs
```shell
$ python3 scripts/10motif.py 
```

### Evaluating paired-motifs
```shell
$ python3 scripts/paired.py
```

### Evaluating real data
```shell
$ python3 scripts/real_data.py data/sample/sample.fasta
```

### Run GMM
```shell
$ python3 scripts/gmm.py \
    data/sample/sample.fasta \
    data/sample/cnn_phmm_vae.mdl
```

### Run BO
```shell
$ python3 scripts/bo.py \
    data/external/A_4R.fastq \
    results/real_data/A_best.mdl \
    results/real_data/A_evaled.csv
```


## Directory structure
```
.
├── data
│   ├── external
│   ├── generated
│   │   ├── 10motifs
│   │   └── paired_motif
│   └── sample
├── results
│   ├── generated
│   │   ├── 10motifs
│   │   └── paired_motif
│   └── real_data
├── scripts
└── src
    ├── data
    ├── models
    └── visualization
```