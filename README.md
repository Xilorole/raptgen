# Supplementary code for "RaptGen: A variational autoencoder with profile hidden Markov model for generative aptamer discovery"

## Tested environment

* Ubuntu == 18.04
* python == 3.7
* pytorch == 1.5.0
* cuda == 10.2

For other requirements, see [Pipfile](Pipfile). Also We verified that the codes are runnable in the provided Docker environment (see [Dockerfile](Dockerfile)). Built image is available at [`natuski/raptgen-gpu`](https://hub.docker.com/repository/docker/natuski/raptgen-gpu) on docker hub. The requirements are installable using [pipenv](https://pipenv.pypa.io/en/latest/) with;

```shell
pipenv install
```

The install time was about 10 minutes on MacbookPro 2020 Core i5 16G. You may also need to install `cairo` library to generate profile hmm image. For mac OS X, it can be installed by `brew install cairo && brew install pango`. For Ubuntu `sudo apt-get install -y libcairo2` would work.

## Quickstart

All scripts has `--help` command that print the options and the arguments if required. For example,

```text
% python scripts/multiple.py --help 
Usage: multiple.py [OPTIONS]

  run experiment with multiple motif

Options:
  --n-motif INTEGER       the number of motifs  [default: 10]
  --n-seq INTEGER         the number of the sequence to generate  [default:
                          10000]

  --seed INTEGER          seed for seqeunce generation reproduction  [default:
                          0]

  --error-rate FLOAT      the ratio to modify sequence  [default: 0.1]
  --epochs INTEGER        the number of training epochs  [default: 1000]
  --threshold INTEGER     the number of epochs with no loss update to stop
                          training  [default: 50]

  --use-cuda / --no-cuda  use cuda if available  [default: True]
  --cuda-id INTEGER       the device id of cuda to run  [default: 0]
  --save-dir PATH         path to save results  [default:
                          out/simlulation/multiple]

  --reg-epochs INTEGER    the number of epochs to conduct state transition
                          regularization  [default: 50]

  --help                  Show this message and exit.  [default: False]
```

Visualized logs looks like
```
% python3 scripts/real.py data/sample/sample.fasta 
saving to /Users/niwn/raptgen/out/real
reading fasta format sequence
adapter info not provided. estimating value
estimated forward adapter len is 5 : AAAAA
estimated reverse adapter len is 5 : GGGGG
filtering with : AAAAA(5N)-20N-GGGGG(5N)
experiment name : 20211128_210830338899
# of sequences -> 100

[1] 139 itr  26.23 <->  26.99 ( 25.85+  1.14) of _vae.mdl..:  14%|█       | 139/1000 [02:38<16:16,  1.13s/it]
```

The last line indicates the training status. The loss, iteration number, estimated time for training and etc. are shown.

```
[1] 139 itr  26.23 <->  26.99 ( 25.85+  1.14) of _vae.mdl..:  14%|█       | 139/1000 [02:38<16:16,  1.13s/it]
^^^          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^
 |           train & valid (recon+norm.) loss  model filename
 |
number of epochs with no validation loss update.
```


## To evaluate multi-/pair- motif for testing

### Evaluating multi-motifs

To run the experiment with multiple sequence motifs, run;

```shell
python3 scripts/multiple.py
```

This outputs models ({{model}}.mdl) and its training result ({{model}}.csv) into specified folder (default is out/simlulation/multiple). A single run takes approximately 20 hours on Tesla V100 GPU.

### Evaluating paired-motifs

To run the experiment with paired sequence motifs, run;

```shell
python3 scripts/paired.py
```

The default saving folder is out/simlulation/paired. A single run takes approximately 10 hours on Tesla V100 GPU.

## To evaluate real data

In order to run raptgen with your own sequence files, you have to run `real.py` which trains the model which encode sequence into representation vector.

### Evaluating real data

To run the experiment with sequence files, run;

```shell
python3 scripts/real.py data/sample/sample.fasta
```

`.fa`, `.fasta`, and `.fastq` files are automatically processed. The default saving folder is `out/simlulation/real`. The runtime depends on the sequence length and number of unique sequences. The output of this procedure are the followings;

* trained model : `[MODEL_NAME].mdl`, such as `cnn_phmm_vae.mdl`
* model loss transition: `[MODEL_NAME].csv`, such as `cnn_phmm_var.csv`

Model which calculates probability using [polytope model](https://en.wikipedia.org/wiki/Polytope_model) is experimentally implemented. It is about 10x faster when random region length is around 100. To use polytope model, use `real_fast.py`.

### Utilizing trained encoder and decoder

To embded and reconstruct sequence, use `encode.py` and `decode.py`. `encode.py` gives its representation vector.

### Run GMM

To select the center of the GMM populations, run;

```shell
python3 scripts/gmm.py \
  data/sample/sample.fasta \
  data/sample/cnn_phmm_vae.mdl
```

this will output top 10 sequence to specified directory (default out/gmm/gmm_seq.csv).

### Run BO

To conduct multipoint Bayesian optimization, run;

```shell
python3 scripts/bo.py \
  data/real/A_4R.fastq \
  results/real/A_best.mdl \
  results/real/A_evaled.csv
```

The evaluates seuqneces should hold random region only and each row should be written in  `[string],[value]` format.

```text
AACGAGAGATGGTAGACCTATCTTTTAGCC,79.0
GTAGAGATTCTGAGGGTTCTCCTGCTATA,107.1
TTTTATAAAAAAGTGTTTAAAAAAGATTCA,-3.6
...
```

The result contains the sequence to be evaluated, the position of the motif embedding, and the embedding of the most probable sequence (`re_`).

```
% cat out/bo/bo_seq.csv
bo_index,seq,x,y,re_x,re_y
0,GTAGAGATTCTGAGGGTTCTCCTGTTGACC,1.53,-0.13,1.60,-0.50
1,GTAGAGATTCTGAGGGTTCTCCTGTTGCCA,1.56,-0.58,1.62,-0.47
```

## Directory structure

```text
.
├── data
│   ├── real
│   ├── sample
│   └── simulation
│       ├── multiple
│       └── paired
├── results
│   ├── real
│   └── simulation
│       ├── multiple
│       └── paired
├── scripts
└── src
    ├── data
    ├── models
    └── visualization
```
