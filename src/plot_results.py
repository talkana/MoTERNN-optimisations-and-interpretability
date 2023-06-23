import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from simulation_scripts.constants import INDEX_TO_MODE


def parse_args():
    parser = argparse.ArgumentParser(description='main script of MoTERNN')
    parser.add_argument('--input_path', required=False, default="evaluation_results.csv", help='path to the directory with full evaluation results')
    parser.add_argument("--output_path", required=False, default="accuracy_plot.png", help='path of the output plot')
    return parser.parse_args()


def save_accuracy_plot(df, outfile):
    df = df[df["Actual"].isin([1, 2])]
    correct_df = df[df['Correct'] == True]
    group_categories = ['Leaves underneath', 'Actual']
    grouped_correct_df, grouped_full_df = correct_df.groupby(group_categories), df.groupby(group_categories)
    accuracy_df = grouped_correct_df.size().div(grouped_full_df.size()).reset_index(name='Accuracy')
    accuracy_df['mode of evolution'] = accuracy_df['Actual'].apply(lambda x: INDEX_TO_MODE[x])
    accuracy_df["Size of the tree (leaves)"] = accuracy_df["Leaves underneath"]
    sns.set_palette("bright")
    sns.lineplot(data=accuracy_df, x='Size of the tree (leaves)', y='Accuracy', hue='mode of evolution')
    plt.title("Accuracy of model predictions on simulation subtrees")
    plt.savefig(outfile)


def main():
    args = parse_args()
    data = pd.read_csv(args.input_path)
    save_accuracy_plot(data, args.output_path)


if __name__ == "__main__":
    main()
