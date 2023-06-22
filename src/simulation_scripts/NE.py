from ete3 import Tree
import numpy as np
import random

from constants import MODE_TO_INDEX
from utils import add_root_children, add_names_and_distances, add_simulation_type_labels

np.random.seed(0)
random.seed(0)


def ne(n=20, root=None):
    if n == 0 or (n == 1 and root is not None):
        return root

    if root is None:
        root = Tree(name="root")
        root.add_features(lb=0, ub=1)
        create_names = True
    else:
        create_names = False

    root.lb = 0
    root.ub = 1

    alpha = 1e4
    beta = 1e4
    us_ = np.random.uniform(0, 1, n - 1)
    bs_ = np.random.beta(float(alpha + 1), float(beta + 1), n - 1)

    j = 0
    while j < n - 1:
        if j == 0:
            add_root_children(root, bs_[j])
        else:
            for leaf in root:
                if leaf.lb < us_[j] < leaf.ub:
                    # split the leaf into two child nodes
                    left = leaf.add_child()
                    left.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb) * bs_[j])
                    right = leaf.add_child()
                    right.add_features(lb=leaf.lb + (leaf.ub - leaf.lb) * bs_[j], ub=leaf.ub)
                    break
        j += 1
    add_simulation_type_labels(root, simtype=MODE_TO_INDEX["neutral"])
    if not create_names:
        return root
    else:
        return add_names_and_distances(root)


if __name__ == "__main__":
    tree, n_muts = ne(n=200)
    print(tree)
    print(n_muts)
    print(len(tree))
