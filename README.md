# Supplementary code for "RaptGen: A variational autoencoder with profile hidden Markov model for generative aptamer discovery"

## Tested environment

* Ubuntu == 18.04
* python == 3.7
* pytorch == 1.5.0
* cuda == 10.2

For other requirements, see [Pipfile](Pipfile). Also We verified that the codes are runnable in the provided Docker environment (see [Dockerfile](Dockerfile)). Built image is available at [`natuski/raptgen-gpu`](https://hub.docker.com/repository/docker/natuski/raptgen-gpu) on docker hub. The requirements are installable using [pipenv](https://pipenv.pypa.io/en/latest/) with;

```shell
% pipenv install
```

The install time was about 10 minutes on MacbookPro 2020 Core i5 16G. You may also need to install `cairo` library to generate profile hmm image. For mac OS X, it can be installed by `brew install cairo && brew install pango`. For Ubuntu, `sudo apt-get install -y libcairo2` would work.

## Quickstart

All scripts have `--help` command that prints the options and the arguments if required. For example,

```shell
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

Visualized train logs look like;
```shell
% python3 scripts/real.py data/sample/sample.fasta 
saving to /Users/niwn/raptgen/out/real
reading fasta format sequence
adapter info not provided. estimating value
estimated forward adapter len is 5 : AAAAA
estimated reverse adapter len is 5 : GGGGG
filtering with : AAAAA(5N)-20N-GGGGG(5N)
experiment name : 20211128_210830338899
# of sequences -> 100

[1] 139 itr  26.2 <->  26.9 (25.8+ 1.1) of _vae.mdl..:  14%|█    | 13/100 [02:38<16:16,  11s/it]
```

The last line indicates the training status. The loss, iteration number, estimated time for training, etc., are shown.

```
[1] 139 itr  26.2 <->  26.9 (25.8+ 1.1) of _vae.mdl..:  14%|█    | 13/100 [02:38<16:16,  11s/it]
^^^          ^^^^      ^^^^^^^^^^^^^^^^    ^^^^^^^^^^   ^^^        ^^^^^^  ^^^^^^^^^^^   ^^^^^^   
(1)          (2)             (3)              (4)       (5)         (6)        (7)        (8)
```

1. the number of epochs with no validation loss update.
2. train loss
3. valid (recon+norm.) loss
4. model name
5. training progress
6. number of iteration
7. elapsed time / estimate time of training
8. training speed

## To evaluate real data

To run raptgen with your sequence files, you have to run `real.py`, which trains the model which encodes sequence into representation vector.

### Train RaptGen using real data

To run the experiment with sequence files, run;

```shell
% python3 scripts/real.py data/sample/sample.fasta
```

`.fa`, `.fasta`, and `.fastq` files are automatically processed. The default saving folder is `out/simlulation/real`. The runtime depends on the sequence length and number of unique sequences. The output of this procedure is the followings;

* trained model : `[MODEL_NAME].mdl`, such as `cnn_phmm_vae.mdl`
* model loss transition: `[MODEL_NAME].csv`, such as `cnn_phmm_var.csv`

### Encode sequence to achieve latent representation

To embed the sequence, use `encode.py`, which input sequences and trained model and output sequences' representation vector. While the VAE model encodes the sequence into the latent space in the form of distribution, the output representation vector is the center of this distribution. 

Run;

```shell
% python3 scripts/encode.py \
    data/sample/sample.fasta \
    results/simulation/multiple/cnn_phmm_vae.mdl \
```

This will output sequences' representation vector in the following format;

```csv
index,seq,dim1,dim2
0,CGACATGGGCCGCCCAAGGA,0.14,0.08
1,GCGTACCGTAAATCTGTCGG,0.10,0.03
...
```

The default saving path is `out/encode/embed_seq.csv`.
### Decode latent point to most_probable sequence

To reconstruct sequence from the latent space, use `decode.py`. Given the model parameters and data points, the raptgen model would sample the most probable sequence from the derived profile HMM. Note that the model length has to be explicitly passed to the script to initialize the model.

```shell
% python3 scripts/decode.py \
    out/encode/embed_seq.csv \
    results/simulation/multiple/cnn_phmm_vae.mdl \
    20
```

This will input csv with the identifier columns followed by dimension info;

```
index,dim1,dim2
0,0.14,0.08
1,0.1,0.03
...
```

and output reconstructed model and log probability of the sequence in the following format;

```
index,dim1,dim2,pattern,maximum_probable_sequence,log_proba
0,0.14,0.08,*C*T*ATCCCGCCCC,ACGTGATCCCGCCCC,-17.602188110351562
1,0.1,0.03,*C*T*ATCCCGCTGC,ACATGATCCCGCTGC,-16.477264404296875
...
```

The default saving path is `out/decode/decode_output.csv`.

### Run GMM

To select the center of the GMM populations, run;

```shell
% python3 scripts/gmm.py \
    data/sample/sample.fasta \
    data/sample/cnn_phmm_vae.mdl
```

This will output the top 10 sequences to a specified directory (default out/gmm/gmm_seq.csv).

### Run BO

To conduct multipoint Bayesian optimization, run;

```shell
% python3 scripts/bo.py \
    data/real/A_4R.fastq \
    results/real/A_best.mdl \
    results/real/A_evaled.csv
```

The evaluates sequences should only hold the random region, and each row should be written in  `[string],[value]` format.

```text
AACGAGAGATGGTAGACCTATCTTTTAGCC,79.0
GTAGAGATTCTGAGGGTTCTCCTGCTATA,107.1
TTTTATAAAAAAGTGTTTAAAAAAGATTCA,-3.6
...
```

The result contains:
* The sequence is to be evaluated.
* The position of the motif embedding.
* The embedding of the most probable sequence (`re_`).

```shell
% cat out/bo/bo_seq.csv
bo_index,seq,x,y,re_x,re_y
0,GTAGAGATTCTGAGGGTTCTCCTGTTGACC,1.53,-0.13,1.60,-0.50
1,GTAGAGATTCTGAGGGTTCTCCTGTTGCCA,1.56,-0.58,1.62,-0.47
```

## To evaluate multi-/pair- motif for testing

### Evaluating multi-motifs

To run the experiment with multiple sequence motifs, run;

```shell
% python3 scripts/multiple.py
```

This outputs models (`[MODEL_NAME].mdl`) and its training result (`[MODEL_NAME].csv`) into specified folder (default is out/simlulation/multiple). A single run takes approximately 20 hours on Tesla V100 GPU.

### Evaluating paired-motifs

To run the experiment with paired sequence motifs, run;

```shell
% python3 scripts/paired.py
```

The default saving folder is out/simlulation/paired. A single run takes approximately 10 hours on Tesla V100 GPU.



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
