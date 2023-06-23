import argparse
import numpy as np
import random
import pandas as pd
import os

from PE import pe
from LE import le
from NE import ne
from BE import be
from mutation_assigner import assign

np.random.seed(0)
random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script generates simulated trees for training MoTERNN')
    parser.add_argument('-lb', '--lb', help='minimum number of cells for each phylogeny', default=20, type=int)
    parser.add_argument('-ub', '--ub', help='maximum number of cells for each phylogeny', default=100, type=int)
    parser.add_argument('-nloci', '--nloci', help='number of loci in the genotype profiles', default=3375, type=int)
    parser.add_argument('-nsample', '--nsample', help='number of datapoints generated for each mode of evolution',
                        default=4000, type=int)
    parser.add_argument('-dir', '--dir', help='destination directory to save the simulated data',
                        default="./trees_dir/")
    parser.add_argument('-seed', '--seed', help='random seed', default=0, type=int)
    args = parser.parse_args()
    print(f"input arguments {args}")
    # minimum number of single-cells
    c_lb = args.lb
    # maximum number of single-cells
    c_ub = args.ub
    # the number of loci
    n_loci = args.nloci
    # specify the name of the directory
    target_dir = args.dir
    # specify the number of trees that will be generated for each class
    num_trees = args.nsample
    # set the random seeds of numpy and random
    np.random.seed(args.seed)
    random.seed(args.seed)
    counter = 0
    print(f"simulating {num_trees} trees ...")
    try:
        # Create target Directory
        os.mkdir(target_dir)
        print("Directory ", target_dir, " Created ")
    except FileExistsError:
        print("Directory ", target_dir, " already exists")

    # generate trees for PE model
    print("generate trees for PE model...")
    for i in range(num_trees):

        outname = f"{target_dir}tree{i}_0.nw"
        seq_file = f"{target_dir}seq{i}_0.csv"
        n_cells = random.randint(c_lb, c_ub)
        tree, n_muts = pe(n=n_cells)
        print(f"mode: PE, tree index: {i + 1}, number of nodes in the generated tree: {len(tree)}")
        if n_muts <= n_loci:
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)
        else:
            while n_muts > n_loci:
                print("number of mutations exceeds the number of loci")
                print("try again ...")
                tree, n_muts = pe(n=n_cells)
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)

    # generate trees for NE model
    print("generate trees for NE model...")
    for i in range(num_trees):

        outname = f"{target_dir}tree{i}_1.nw"
        seq_file = f"{target_dir}seq{i}_1.csv"
        n_cells = random.randint(c_lb, c_ub)
        tree, n_muts = ne(n=n_cells)
        print(f"mode: NE, tree index: {i + 1}, number of nodes in the generated tree: {len(tree)}")
        if n_muts <= n_loci:
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)
        else:
            while n_muts > n_loci:
                print("number of mutations exceeds the number of loci")
                print("try again ...")
                tree, n_muts = ne(n=n_cells)
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)

    # generate trees for LE model
    print("generate trees for LE model...")
    for i in range(num_trees):

        outname = f"{target_dir}tree{i}_2.nw"
        seq_file = f"{target_dir}seq{i}_2.csv"
        n_cells = random.randint(c_lb, c_ub)
        tree, n_muts = le(n=n_cells)
        print(f"mode: LE, tree index: {i + 1}, number of nodes in the generated tree: {len(tree)}")
        if n_muts <= n_loci:
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)
        else:
            while n_muts > n_loci:
                print("number of mutations exceeds the number of loci")
                print("try again ...")
                tree, n_muts = le(n=n_cells)
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)

    # generate trees for BE model
    print("generate trees for BE model...")
    for i in range(num_trees):

        outname = f"{target_dir}tree{i}_3.nw"
        seq_file = f"{target_dir}seq{i}_3.csv"
        n_cells = random.randint(c_lb, c_ub)
        tree, n_muts = be(n=n_cells)
        print(f"mode: BE, tree index: {i + 1}, number of nodes in the generated tree: {len(tree)}")
        if n_muts <= n_loci:
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)
        else:
            while n_muts > n_loci:
                print("number of mutations exceeds the number of loci")
                print("try again ...")
                tree, n_muts = be(n=n_cells)
            counter += 1
            tree, seq_dict = assign(t=tree, n_loci=n_loci)
            tree.write(features=["simtype"], outfile=outname)
            df = pd.DataFrame.from_dict(seq_dict)
            df.to_csv(seq_file)

    print("done!")
    print(f"total number of generated trees: {counter}")
    print(f"the training data are stored at {os.path.abspath(target_dir)}")
