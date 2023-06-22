import numpy as np


def add_simulation_type_labels(root, simtype):
    for v in root.traverse(strategy="levelorder"):
        if "simtype" not in v.features:
            v.add_features(simtype=simtype)


def add_names_and_distances(root):
    k = 1
    for v in root.traverse(strategy="levelorder"):
        v.name = "Taxon" + str(k)
        k += 1
    # assign branch lengths to all branches
    n_mutations = 0
    for v in root.get_descendants():
        v.dist = np.random.poisson(5, 1)[0]
        n_mutations += root.dist

    return root, n_mutations


def add_root_children(root, branch_length):
    left = root.add_child()
    left.add_features(lb=0, ub=branch_length)
    right = root.add_child()
    right.add_features(lb=branch_length, ub=1)
