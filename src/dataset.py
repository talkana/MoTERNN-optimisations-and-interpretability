from torch.utils.data import Dataset


class TumorTreesDataset(Dataset):
    def __init__(self, root_folder, indices, nsample):
        self.dataset_indices = indices  # indices of the dataset, eg. train
        self.nsample = nsample
        self.root_folder = root_folder

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset_indices)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        i = self.dataset_indices[index]
        y = i // self.nsample
        tree_nr = i % self.nsample
        assert y < 4
        tree_path, seq_path = f"{self.root_folder}tree{tree_nr}_{y}.nw", f"{self.root_folder}seq{tree_nr}_{y}.csv"
        return tree_path, seq_path, y
