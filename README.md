# MoTERNN - optimisations and interpretability: final project for Modeling of Complex Biological Systems 
## Description
There are four major evolutionary modes that have been proposed in cancer patients. Classifying a tumor into one of these four modes is important for diagnosis and treatment purposes. A paper by Edrisi et al (https://doi.org/10.1101/2022.08.21.504710) introduced a recursive neural network model, called MoTERNN, which predicts the mode of evolution from a reconstructed phylogenetic network of tumor.

This repository contains an updated version of the original MoTERNN scripts. The training and evaluation scripts contain computational optimisations that reduce the peak RAM usage from 59.94 GB to 6.64 GB on the simulated training data used by Edrisi et al. The repository also contains additional scripts to evaluate the model and its interpretability.
## Description of the directories
This repository contains the PyTorch implementation of MoTERNN and the scripts used for generating simulated data. Also, we have provided the trained model and the real data that we applied our method on in the study. Specifically we have:
- `src`: contains all the scripts for training the RNN model and simulating the training data.
- `data`: contains the real dataset including the phylogenetic tree (in Newick format) and the csv file of the genotypes at the leaves. 

The implementation of Recursive Neural Network in this project was adopted from https://github.com/mae6/pyTorchTree. To reporduce the results presented in the paper, please follow the instructions below in [Reproducibility](https://github.com/NakhlehLab/MoTERNN#reproducibility).

## Reproducibility
  1. ### Simulation of the training data
     Download this package, unzip it, and navigate to the main directory named `src`. To make sure `generator.py` works and see the arguments, run:
     ```
     python generator.py --help
     ```
     which shows the following message:
     ```
     This script generates simulated trees for training MoTERNN
     
     optional arguments:
     -h, --help            show this help message and exit
     -lb LB, --lb LB       minimum number of cells for each phylogeny
     -ub UB, --ub UB       maximum number of cells for each phylogeny
     -nloci NLOCI, --nloci NLOCI
                        number of loci in the genotype profiles
     -nsample NSAMPLE, --nsample NSAMPLE
                        number of datapoints generated for each mode of evolution
     -dir DIR, --dir DIR   destination directory to save the simulated data
     -seed SEED, --seed SEED
                        random seed
     ```
     To generate simulated data, run `generator.py`, run the following command:
     ```
     python generator.py -dir ./trees_dir/ -nsample 4000 -lb 20 -ub 100 -seed 0 -nloci 3375
     ```
     This will create a directory named `trees_dir` containing 16000 pairs of .nw and .csv files for each of the four modes of evolution (4000 datapoints for each mode, `-nsample 4000`), on 3375 loci (`-nloci 3375`) with the number of cells varying between 20 (`-lb 20`) and 100 (`-ub 100`), with random seed 0 (`-seed 0`).
   2. ### Running MoTERNN
      To run MoTERNN, use the code named `RNN.py` in `src` directory. First, to make sure the code works, run:
      ```
      python RNN.py --help
      ```
      which will show the following message:
      ```
      main script of MoTERNN

      optional arguments:
        -h, --help            show this help message and exit
        -dir DIR, --dir DIR   path to the directory of the simulated data
        -test TEST, --test TEST
                              fraction of data (in percent) to be selected as test data
        -val VAL, --val VAL   number of datapoints in validation set
        -newick NEWICK, --newick NEWICK
                              path to the real data phylogeny in newick format
        -seq SEQ, --seq SEQ   path to the csv file containing the genotypes of the real data
        -dim DIM, --dim DIM   embedding dimension for the encoder network
        -nsample NSAMPLE, --nsample NSAMPLE
                              number of datapoints generated for each mode of evolution (it must match the same
                              argument used in the generator)
        -seed SEED, --seed SEED
                              random seed
        -nloci NLOCI, --nloci NLOCI
                              number of loci in the genotype profiles (it must match the same arguemnt used in the
                              generator)
      ```
      Now, run the code on the generated data as in the following example:
      ```
      python RNN.py -nsample 4000 -dim 256 -dir ./trees_dir/ -test 0.25 -val 100 -newick ./phylovar.nw -seq ./phylovar_seq.csv -seed 0 -nloci 3375
      ```
      The above command runs the code assuming there are 4000 datapoints for each of the four classes (`-nsample 4000`), and they are stored in `./trees_dir/` directory (`-dir ./trees_dir/`); the encoder network, maps the data into a shared space of size 256 (`-dim 256`); the test set contains 25% of the entire dataset, selected randomly (`-test 0.25`). The validation set contains 100 datapoints chosen randomly (`-val 100`); the topology of the real biological phylogeny in the form of newick string and the genotype sequences are stored in `./phylovar.nw` and `./phylovar_seq.csv`, respectively (they are provided in `data` directory of this repository); the random seed is set to 0 (`-seed 0`), and the number of loci in the real and generated data is 3375 (`-nloci 3375`). 
      At the end of training, the output of the code run with the above settings will be as follows:
      ```
      final accuracy of the model on the training set: 1.0
      final accuracy of the model on the test set: 1.0
      final accuracy of the model on the validation set: 1.0
      prediction on real tree: Punctuated mode
      training was done in 684.0261344909668 seconds
      the trained model was saved at /home/mae6/evolution_modes/repo/moternn.pt
      ```
      which shows the accuracy of training, test, and validation sets in addition to the prediction on the real data. The trained model of this example is provided in this repository at `data` directory named `moternn.pt`
   3. ### Evaluating the trained model
      To evaluate the trained model and apply it on your data (e.g. the phylogeny from the TNBC data), navigate to `src` directory, then run `eval.py` using the following command:
      ```
      python eval.py -model ./moternn.pt -newick ./phylovar.nw -seq ./phylovar_seq.csv -nloci 3375 -dim 256 -seed 0
      ```
      Here, the argument `-model ./moternn.pt` points to the path to the trained model. The rest of the arguments are defined as in [Running MoTERNN](https://github.com/NakhlehLab/MoTERNN/blob/main/README.md#running-moternn). The output of the above command will be:
      ```
      parsing the input data including real and simulated ...
      parsing the real data...
      parsing all data took 0.016454696655273438 seconds
      prediction on real tree: Punctuated mode
      ```
