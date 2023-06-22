from ete3 import Tree
import numpy as np
import random
from NE import ne
from utils import add_names_and_distances, add_root_children, add_simulation_type_labels
from constants import MODE_TO_INDEX

np.random.seed(0)
random.seed(0)


def le(n=20, root=None):
    if n == 0:
        print("ERROR: the number of nodes is zero!")
        return None
    # parameters for the backbone tree
    alpha = 1e4
    beta = 1e-4
    if root is None:
        root = Tree(name="root")
        root.add_features(lb=0, ub=1)
        create_names = True
    else:
        create_names = False
    # divide the total unmber of speciations into two
    # one being 2/3 and the other roughly 1/3
    if n <= 3:
        s1 = n - 1
        s2 = 0
    else:
        q, right = divmod(n - 1, 3)
        s1 = 2 * q
        s2 = q + right

    # step 1: split the root into left and right child with respective labels
    if n != 1:
        us_ = np.random.uniform(0, 1, n - 1)
        bs_ = np.random.beta(float(alpha + 1), float(beta + 1), n - 1)
        add_root_children(root, bs_[0])
        # step 2: search for the leaf node whose interval covers us_[j], then split it into two child nodes
        # one with [0,b1b2] interval and the other with [b1b2,1]as its interval
        # step 3: continue for s1 steps
        j = 1
        while j < s1:
            for leaf in root:
                if leaf.lb < us_[j] < leaf.ub:
                    # split the leaf into two child nodes
                    left = leaf.add_child()
                    left.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb) * bs_[j])
                    right = leaf.add_child()
                    right.add_features(lb=leaf.lb + (leaf.ub - leaf.lb) * bs_[j], ub=leaf.ub)
                    j += 1
                    break
        node_, d = root.get_farthest_leaf()
        ne(n=s2 + 1, root=node_)

    add_simulation_type_labels(root, simtype=MODE_TO_INDEX["linear"])
    return add_names_and_distances(root) if create_names else root


if __name__ == "__main__":
    tree, n_muts = le(n=20)
    # tree.write(format=3, outfile="linear_example.nw")
    print(tree)
    print(n_muts)
    print(len(tree))







