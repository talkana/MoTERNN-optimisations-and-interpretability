import copy
import random
random.seed(0)


def assign(t, n_loci):
    ###################################
    # adding the mutations to the nodes
    # keep the indices of all mutations
    muts_idx = []
    # list of all positions
    all_idx = [i for i in range(n_loci)]
    # sequence of the root
    seq = [0 for _ in range(n_loci)]
    # assign the sequence to the root
    t.add_features(seq=seq)
    # traverse tree from the root to the leaves
    for node in t.traverse(strategy="levelorder"):
        if not node.is_root():
            choices_idx = list(set(all_idx) - set(muts_idx))
            node_seq = copy.deepcopy(node.up.seq)
            if node.dist > 0:
                idx = random.sample(choices_idx, int(node.dist))
                for x in idx:
                    node_seq[x] = 1
                    muts_idx.append(x)
            node.add_features(seq=node_seq)
    seq_dict = {}
    for node in t.traverse():
        if node.is_leaf():
            seq_dict[node.name] = node.seq
    return t, seq_dict




