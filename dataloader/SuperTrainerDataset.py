from torch.utils.data import Dataset


class SuperTrainerDataset(Dataset):
    def __init__(self):
        super(SuperTrainerDataset, self).__init__()
        self.length_list = [0]
        self.limit_indices = [0]
        self.datasets = list()

    def __getitem__(self, item):
        for idx, limit in enumerate(self.limit_indices):
            if item < limit:
                return self.datasets[idx - 1][item - self.limit_indices[idx - 1]]
        assert False, "Index greater than dataset size"

    def __len__(self):
        return sum(self.length_list)

    def add_dataset(self, dataset):
        self.length_list.append(len(dataset))
        self.limit_indices.append(sum(self.length_list))
        self.datasets.append(dataset)



