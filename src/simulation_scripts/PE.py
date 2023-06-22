from ete3 import Tree
from NE import ne
from constants import MODE_TO_INDEX
from utils import add_simulation_type_labels

import numpy as np
import random

np.random.seed(0)
random.seed(0)


def pe(n=10):
    # number of cells
    if n == 0:
        print("ERROR: the number of nodes is zero!")
        return None

    K = random.sample([2, 3], 1)[0]
    c_ = np.random.multinomial(n - 1 - 2 * K, [1 / K] * K, size=1)[0]
    t = Tree(name="root")
    normal = t.add_child()
    normal.add_features(lb=0, ub=1)
    tumor = t.add_child()
    tumor.add_features(lb=0, ub=1)

    ne(n=K, root=tumor)
    for i, leaf in enumerate(tumor.get_leaves()):
        ne(n=c_[i] + 2, root=leaf)

    # create alphabetical names for the leaves and internal nodes
    k = 0
    for node in t.traverse(strategy="levelorder"):
        node.name = "Taxon" + str(k)
        k += 1
    # assign branch lengths to all branches
    n_mutations = 0
    # the normal cell has zero distance to the root
    normal.dist = 0
    n_mutations += normal.dist
    # sample the number of mutations on the long trunk of the tree from
    # a Poisson distribution with lambda = 1000
    tumor.dist = np.random.poisson(100, 1)[0]
    # tumor.dist = random.randint(100, 1000)
    n_mutations += tumor.dist
    # sample the number of mutations for clonal branches from
    # a Poisson distribution with lamda = 5
    for node in tumor.get_descendants():
        node.dist = np.random.poisson(5, 1)[0]
        n_mutations += node.dist
    add_simulation_type_labels(t, simtype=MODE_TO_INDEX["punctuated"])
    return t, n_mutations


if __name__ == "__main__":
    tree, n_muts = pe(n=20)
    print(tree)
    print(n_muts)
    print(len(tree))
