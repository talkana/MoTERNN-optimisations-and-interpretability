import numpy as np
from ete3 import Tree
import random
from NE import ne, add_names_and_distances, add_simulation_type_labels
from LE import le
from constants import MODE_TO_INDEX

np.random.seed(0)
random.seed(0)


def be(n=20):
    if n == 0:
        print("ERROR: the number of nodes is zero!")
        return None

    # neutral model until k leaves reached
    t = Tree(name="root")
    K = random.sample([2, 3, 4], 1)[0]
    ne(n=K, root=t)
    # linear model for each branch
    # minimum number of cells for each branch should be 2, 2*K nodes are subtracted from total
    c_ = np.random.multinomial(n - 2 * K, [1 / K] * K, size=1)[0]
    for i, leaf in enumerate(t.get_leaves()):
        le(n=c_[i] + 2, root=leaf)
    add_simulation_type_labels(t, simtype=MODE_TO_INDEX["branching"])
    return add_names_and_distances(t)


if __name__ == "__main__":
    tree, n_muts = be(n=30)
    print(tree)
    print(n_muts)
    print(len(tree))
