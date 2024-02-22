from torch.utils.data import DataLoader


class MetricDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        batch = [self.dataset[i] for i in range(start_idx, min(end_idx, len(self.dataset)))]

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        return batch
