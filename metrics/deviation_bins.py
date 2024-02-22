from dataloader.SwingDatasets import HourlyDataset
from dataloader.MetricDataloader import MetricDataloader
import torch


class DeviationEvaluator:
    def __init__(self, model, dataset, bin_size, limit_error):
        self.model = model
        self.orig_model_device = next(model.parameters()).device
        self.dataset = dataset
        self.dataloader = MetricDataloader(dataset, batch_size=1, shuffle=False, collate_fn=HourlyDataset.get_collate_fn())
        self.limit_error = limit_error
        self.bin_dict = self._create_bins(bin_size, limit_error)

    def evaluate(self):
        # reset the bin dict
        for key, value in self.bin_dict.items():
            self.bin_dict[key] = 0

        self.model.to("cpu")
        self.model.eval()
        with torch.no_grad():
            for inputs in self.dataloader:
                _, _, label, _, _ = inputs
                label = label.numpy()[0][1]     # get rid of BS and get the value itself
                out, _ = self.model(inputs)
                out = out.numpy()[0]            # get rid of BS
                distance = abs(out - label)

                for key, value in self.bin_dict.items():
                    if key > distance:
                        self.bin_dict[key] += 1
                        break

        self.model.to(self.orig_model_device)

    def get_bin_dict(self, format):
        if format == "number":
            return self.bin_dict
        elif format == "percentage":
            total_trials = sum(self.bin_dict.values())
            new_dict = dict()
            for key in self.bin_dict:
                new_dict[key] = self.bin_dict[key] / total_trials
            return new_dict
        else:
            assert False

    def _create_bins(self, bin_size, limit_error):
        bin_dict = dict()
        cur_limit = bin_size
        while cur_limit < limit_error:
            bin_dict[cur_limit] = 0
            cur_limit += bin_size

        bin_dict[999999] = 0
        return bin_dict


