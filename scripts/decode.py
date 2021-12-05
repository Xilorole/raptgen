# the position are meant to be 2 dim.
# csv file to read the target generate position should be written in the follow¥ing format
# 
# i, x, y
# 0, 0.24, 0.33
# 1, 1.0, 0.2
# ...


import logging

import click 
from pathlib import Path

import torch

from raptgen import models
from raptgen.models import CNN_PHMM_VAE
from raptgen.data import SingleRound, Result

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/embed").resolve())

@click.command(help='achieve sequence vector in embedded space.',
    context_settings=dict(show_default=True))
@click.argument("model-path", help="the path of the saved model parameters", type=click.Path(exists = True))
@click.argument("pos-path", help="embedded point to generate profile HMM model", type=click.Path(exists=True))
@click.argument("seq-path", help="the fasta/fastq file used to train model.", type=click.Path(exists=True))
@click.option("--target-len", help="length of the random region of SELEX experiment", type=int)
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--generate-seq", help = "whether or not to create most probable sequence", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
@click.option("--embed-dim", help="the embedding dimension of raptgen model", type=int, default=2)
def main(model_path, pos_path, seq_path, target_len, cuda_id, use_cuda, save_dir, fwd, rev, embed_dim):
    """generate sequences from embedded sequences. Given the position of the sequence,
    the reconstructed profile HMM model is calculated.
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok = True, parents=True)

    # by passing the model sequence path, this can recognize the length of the random region.
    # [TODO] the model should load both parameter and the model structure. update using ONNX format also.
    if not target_len and seq_path:
        experiment = SingleRound(
            path = seq_path,
            forward_adapter = fwd,
            reverse_adapter = rev)
        target_len = experiment.random_region_length

    # the embedded model's default embedding dimension is 2. 
    # if the model is using the fast, use fast model evaluation.
    model = CNN_PHMM_VAE(target_len, embed_size=embed_dim)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))

    result = Result(
        model,
        experiment=experiment,
        path_to_save_results=save_dir,
        load_if_exists=True
    )    

    # turn given position to torch tensor
    # load file and read position

    import pandas as pd
    df = pd.read_csv(pos_path)


    

    model.decoder()
    sequences = experiment.get_filter_passed_sequences(random_only=True)
    embed_x = result.embed_sequences(sequences)
    
    logger.info(f"saving to {save_dir}/embed_seq.csv")
    with open(save_dir/"embed_seq.csv","w") as f:
        f.write("index,seq,dim1,dim2\n")
        for i,(seq,(x1,x2)) in enumerate(zip(sequences, embed_x)):
            logger.info(f"{seq},({x1:.2f},{x2:.2f})")
            f.write(f"{i},{seq},{x1},{x2}\n")

if __name__ == "__main__":
    Path("./.log").mkdir(parents=True, exist_ok=True)
    formatter = '%(levelname)s : %(name)s : %(asctime)s : %(message)s'
    logging.basicConfig(
        filename='.log/logger.log',
        level=logging.DEBUG,
        format=formatter)
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    main()
