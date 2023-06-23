import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from simulation_scripts.constants import INDEX_TO_MODE
from dataset import TumorTreesDataset
from tree_parsing import parse_tree

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# identify the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# this code is adopted from https://github.com/aykutfirat/pyTorchTree

class RNN(nn.Module):
    def __init__(self, sequence_size, embedSize=100, numClasses=2):
        super(RNN, self).__init__()
        # dictionary for each genomic sequence
        # self.embedding = nn.Embedding(int(vocabSize), embedSize)
        self.embedding = nn.Linear(sequence_size, embedSize)
        # combining the embedding of the children nodes into one embedding
        self.W = nn.Linear(2 * embedSize, embedSize)
        # prediction of the node class given the embedding of a node
        self.projection = nn.Linear(embedSize, numClasses)
        # activation function
        self.activation = F.relu
        # list of class predictions for all the nodes
        self.nodeProbList = []
        # list of node labels
        self.labelList = []

    # this is a recursive function that traverses the tree from the root
    # applies the function recursively on all the nodes

    def traverse(self, node):
        # if the node is a leaf, then only the leaf sequence is considered
        if node.is_leaf():
            currentNode = self.activation(self.embedding(Var(torch.FloatTensor(node.seq)[None, :])))
        # if node is internal, then the embedding of the node will be a function of the children nodes' embeddings
        else:
            currentNode = self.activation(
                self.W(torch.cat((self.traverse(node.get_children()[0]), self.traverse(node.get_children()[1])), 1)))
        if node.is_root():
            # add the class probabilities for the current node into a list
            self.nodeProbList.append(self.projection(currentNode))
            # add the label of the current node into a list
            self.labelList.append(torch.LongTensor([node.label]))
        return currentNode

    def traverse_predict_all(self, node, min_size=4):
        if node.is_leaf():
            currentNode = self.activation(self.embedding(Var(torch.FloatTensor(node.seq)[None, :])))
        else:
            currentNode = self.activation(
                self.W(
                    torch.cat((self.traverse_predict_all(node.get_children()[0]), self.traverse_predict_all(node.get_children()[1])), 1)))

        last_layer = self.projection(currentNode)
        prediction = int(last_layer.max(dim=1)[1])
        actual = node.simtype
        leaves_nr = len(node)
        if leaves_nr > min_size:
            self.nodeProbList.append((prediction, actual, prediction == actual, leaves_nr))
        return currentNode

    def forward(self, x):
        self.nodeProbList = []
        self.labelList = []
        # update the above lists by traversing the given tree
        self.traverse(x)
        self.labelList = Var(torch.cat(self.labelList))
        return torch.cat(self.nodeProbList)

    def getLoss(self, tree):
        # get the probabilities
        nodes = self.forward(tree)
        predictions = nodes.max(dim=1)[1]
        loss = F.cross_entropy(input=nodes, target=self.labelList)
        return predictions, loss, nodes

    def evaluate(self, trees):
        incorrects = []
        incorrect_labels = []
        # calculate the accuracy of the model
        n = correctRoot = 0.0
        for j, (curr_tree_path, curr_seq_path, curr_label) in enumerate(trees):
            curr_tree = parse_tree(curr_tree_path, curr_seq_path, curr_label)
            preds, loss, logits = self.getLoss(curr_tree)
            correct = preds == curr_label
            if not correct:
                incorrects.append(curr_tree)
                incorrect_labels.append(preds)
            correctRoot += correct.squeeze()
            n += 1
        return correctRoot / n, preds, logits, incorrects, incorrect_labels

    def evaluate_full(self, trees):
        is_correct_full = []
        number_of_leaves = []
        predictions_full = []
        actual_full = []
        for j, (curr_tree_path, curr_seq_path, curr_label) in enumerate(trees):
            curr_tree = parse_tree(curr_tree_path, curr_seq_path, curr_label)
            self.nodeProbList = []
            self.traverse_predict_all(curr_tree)
            results = self.nodeProbList
            for result in results:
                prediction, actual, is_correct, depth = result
                is_correct_full.append(is_correct)
                number_of_leaves.append(depth)
                predictions_full.append(prediction)
                actual_full.append(actual)
        return is_correct_full, number_of_leaves, predictions_full, actual_full

# this function passes the input variable to the current device
def Var(v):
    return Variable(v.to(device))


def range_limited_float_type(arg):
    # specifying a range of values for the percentage of test data
    # maximum percentage of test data is 95%
    max_value = 0.95
    # minimum percentage of test data is 5%
    min_value = 0.05
    try:
        arg_value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if arg_value < min_value or arg_value > max_value:
        raise argparse.ArgumentTypeError("The argument must be between < " + str(max_value) + "and > " + str(min_value))
    return arg_value


def partition_dataset(dataset_size, test_size, validation_size):
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    validation_indices = indices[test_size:test_size + validation_size]
    train_indices = indices[test_size + validation_size:]

    return test_indices, validation_indices, train_indices

#################################################
#################### Main #######################
#################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main script of MoTERNN')
    parser.add_argument('-dir', '--dir', help='path to the directory of the simulated data', default="./trees_dir")
    parser.add_argument('-test', '--test', help='fraction of data (in percent) to be selected as test data',
                        default=0.25, type=range_limited_float_type)
    parser.add_argument('-val', '--val', help="number of datapoints in validation set", default=100, type=int)
    parser.add_argument('-newick', '--newick', help="path to the real data phylogeny in newick format",
                        default='./phylovar.nw')
    parser.add_argument('-seq', '--seq', help="path to the csv file containing the genotypes of the real data",
                        default="./phylovar_seq.csv")
    parser.add_argument('-dim', '--dim', help="embedding dimension for the encoder network", default=256, type=int)
    parser.add_argument('-nsample', '--nsample',
                        help="number of datapoints generated for each mode of evolution (it must match the same argument used in the generator)",
                        default=2000, type=int)
    parser.add_argument('-seed', '--seed', help='random seed', default=0, type=int)
    parser.add_argument('-nloci', '--nloci',
                        help='number of loci in the genotype profiles (it must match the same arguemnt used in the generator)',
                        default=3375, type=int)
    parser.add_argument('--evaluate_full', action='store_true', default=False,
                        help="Run evaluation on internal nodes of the trees (and not just the root)")
    parser.add_argument('--eval_path', type=str, default="evaluation_results.csv",
                        help="Path to the file with full evaluation results")
    args = parser.parse_args()

    if args.dir.endswith("/"):
        print("not changing the directory name")
    else:
        args.dir += "/"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print(f"input arguments {args}")
    print("parsing the real data...")
    s = time.time()
    print("Construct datasets ...")
    num_datapoints = 4 * args.nsample
    num_test = int(args.test * num_datapoints)
    num_val = args.val
    merged_dataset = DataLoader(
        TumorTreesDataset(root_folder=args.dir, nsample=args.nsample, indices=list(range(num_datapoints))),
        shuffle=True)
    test_indices, validation_indices, train_indices = partition_dataset(num_datapoints, num_test, num_val)
    test_dataset = DataLoader(TumorTreesDataset(root_folder=args.dir, nsample=args.nsample, indices=test_indices),
                              shuffle=True)
    validation_dataset = DataLoader(
        TumorTreesDataset(root_folder=args.dir, nsample=args.nsample, indices=validation_indices), shuffle=True)
    train_dataset = DataLoader(TumorTreesDataset(root_folder=args.dir, nsample=args.nsample, indices=train_indices),
                               shuffle=True)
    print("start training the model ...")
    model = RNN(sequence_size=args.nloci, embedSize=args.dim, numClasses=4).to(device)
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
    max_epochs = 1
    learning_rate = 1e-4
    wd = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    bestAll = bestRoot = 0.0

    iter_counter = 0
    loss_list = []
    for epoch in range(max_epochs):
        print(f"epoch {epoch}")
        for step, (tree_path, seq_path, label) in enumerate(train_dataset):
            tree = parse_tree(tree_path, seq_path, label)
            # train the model
            model.train()
            predictions, loss, logits = model.getLoss(tree)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5, norm_type=2.0)
            optimizer.step()
            loss_list.append(loss.item())

            # print the iteration index and loss
            print("iteration: {}, loss: {}".format(iter_counter, loss_list[-1]))
            # increment the iteration index
            iter_counter += 1

    ## evaluation of the trained model at the end of training
    dict_ = INDEX_TO_MODE
    model.eval()
    with torch.no_grad():
        # evaluation on the entire training set
        correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(train_dataset)
        print("final accuracy of the model on the training set: {}".format(correctRoot))
        correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(test_dataset)
        print("final accuracy of the model on the test set: {}".format(correctRoot))
        correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(validation_dataset)
        print("final accuracy of the model on the validation set: {}".format(correctRoot))
        correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate([(args.newick, args.seq, 0)])
        print(f"prediction on real tree: {dict_[preds[-1].item()]}")
        if args.evaluate_full:
            is_correct, leaves, predictions, actual = model.evaluate_full(merged_dataset)
            df = pd.DataFrame(
                {"Correct": is_correct, "Leaves underneath": leaves, "Predicted": predictions, "Actual": actual})
            df.to_csv(args.eval_path)
            print(f"Evaluation results were saved at {os.path.abspath(args.eval_path)}")

    print(f"training was done in {time.time() - s} seconds")

    # save the trained model
    target_dir = './moternn.pt'
    torch.save(model.state_dict(), target_dir)
    print(f"the trained model was saved at {os.path.abspath(target_dir)}")
