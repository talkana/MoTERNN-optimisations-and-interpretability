# MoTERNN - optimisations and interpretability: final project for Modeling of Complex Biological Systems 
## Description
There are four major evolutionary modes that have been proposed in cancer patients. Classifying a tumor into one of these four modes is important for diagnosis and treatment purposes. A paper by Edrisi et al (https://doi.org/10.1101/2022.08.21.504710) introduced a recursive neural network model, called MoTERNN, which predicts the mode of evolution from a reconstructed phylogenetic network of tumor.

This repository contains an updated version of the original MoTERNN scripts. The training and evaluation scripts contain computational optimisations that reduce the peak RAM usage from 59.94 GB to 6.64 GB on the simulated training data used by Edrisi et al. The repository also contains additional functions to evaluate the model and its interpretability.

This readme is partially based on the readme from the original repository.
## Reproducibility
  Start by navigating to the 'src' directory and installing the required packages using
  ```
  pip install -r requirements.txt
  ```
  1. ### Simulation of the training data
     To generate simulated data, run `generator.py` with the following command:
     ```
     python generator.py -dir ./trees_dir/ -nsample 4000 -lb 20 -ub 100 -seed 0 -nloci 3375
     ```
     This will create a directory named `trees_dir` containing 16000 pairs of .nw and .csv files for each of the four modes of evolution (4000 datapoints for each mode, `-nsample 4000`), on 3375 loci (`-nloci 3375`) with the number of cells varying between 20 (`-lb 20`) and 100 (`-ub 100`), with random seed 0 (`-seed 0`).
   2. ### Running MoTERNN
      To run MoTERNN on the generated data use the following command:
      ```
      python train.py -nsample 4000 -dim 256 -dir ./trees_dir/ -test 0.25 -val 100 -newick ./phylovar.nw -seq ./phylovar_seq.csv -seed 0 -nloci 3375 --evaluate_full
      ```
      The above command runs the code assuming there are 4000 datapoints for each of the four classes (`-nsample 4000`), and they are stored in `./trees_dir/` directory (`-dir ./trees_dir/`); the encoder network, maps the data into a shared space of size 256 (`-dim 256`); the test set contains 25% of the entire dataset, selected randomly (`-test 0.25`). The validation set contains 100 datapoints chosen randomly (`-val 100`); the topology of the real biological phylogeny in the form of newick string and the genotype sequences are stored in `./phylovar.nw` and `./phylovar_seq.csv`, respectively (they are provided in `data` directory of this repository); the random seed is set to 0 (`-seed 0`), and the number of loci in the real and generated data is 3375 (`-nloci 3375`). The `--evaluate_full` option indicates that the predictions of the model are computed also for subtrees of the simulated phylogenetic trees. The predictions for the subtrees are saved as 'evaluation_results.csv'. The trained model of this example is provided in this repository at `data` directory named `moternn_optimised.pt`.

  3. ### Plotting the results
     
  
  4. ### Using the trained model on new data
      To evaluate the trained model and apply it on your data (e.g. the phylogeny from the TNBC data), navigate to `src` directory, then run the following command:
      ```
      python run.py -model ./moternn.pt -newick ./phylovar.nw -seq ./phylovar_seq.csv -nloci 3375 -dim 256 -seed 0
      ```
      Here, the argument `-model` stores the path of the trained model and the arguments `-seq` and `-newick` store the sequences and the phylogenetic tree respectively. Note that you can find the data used in Edrisi et al. in this repository at `data` directory.
