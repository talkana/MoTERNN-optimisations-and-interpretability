import argparse
import os
import random

import torch

from train import RNN
from simulation_scripts.constants import INDEX_TO_MODE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# identify the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main script of MoTERNN')
    parser.add_argument('-newick', '--newick', help="path to the real data phylogeny in newick format",
                        default='./phylovar.nw')
    parser.add_argument('-seq', '--seq', help="path to the csv file containing the genotypes of the real data",
                        default="./phylovar_seq.csv")
    parser.add_argument('-dim', '--dim', help="embedding dimension for the encoder network", default=256, type=int)
    parser.add_argument('-seed', '--seed', help='random seed', default=0, type=int)
    parser.add_argument('-nloci', '--nloci',
                        help='number of loci in the genotype profiles (it must match the same arguemnt used in the generator)',
                        default=3375, type=int)
    parser.add_argument('-model', '--model', help='path to the trained model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = RNN(sequence_size=args.nloci, embedSize=args.dim, numClasses=4).to(device)
    ## evaluation of the trained model at the end of training
    model.load_state_dict(torch.load(args.model))
    model.eval()
    with torch.no_grad():
        correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate([(args.newick, args.seq, 0)])
        print(f"prediction on real tree: {INDEX_TO_MODE[preds[-1].item()]}")
