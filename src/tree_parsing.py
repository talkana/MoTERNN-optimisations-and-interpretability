import pandas as pd
from ete3 import Tree, parser


def read_ete_tree(newick_f):
    try:
        ete_tree = Tree(newick_f, format=8)
    except parser.newick.NewickError:
        ete_tree = Tree(newick_f)
    return ete_tree


# this function parses an individual tree along with its genotype matrix
def parse_tree(tree_path, seq_path, label=None):
    # read the tree from newick file
    if isinstance(tree_path, tuple):
        tree_path = tree_path[0]
    if isinstance(seq_path, tuple):
        seq_path = seq_path[0]
    tree = read_ete_tree(tree_path)
    # read the sequences from the csv file
    seq_df = pd.read_csv(seq_path, index_col=0)

    for node in tree.traverse(strategy="levelorder"):
        node.add_features(label=int(label))
        if node.is_leaf():
            node.add_features(seq=seq_df[node.name])
    tree.add_features(label=int(label))
    tree.add_features(simtype=int(label))
    return tree
