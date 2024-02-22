import torch
from dataloader.SwingDatasets import HourlyDataset
from dataloader.MetricDataloader import MetricDataloader
from datetime import timedelta


class TradeSetupMetric:
    def __init__(self, model, dataset, limit_days=3):
        self.orig_model_device = next(model.parameters()).device
        self.model = model
        self.dataset = dataset
        self.dataloader = MetricDataloader(dataset, batch_size=1, shuffle=False, collate_fn=HourlyDataset.get_collate_fn())
        self.num_setups = 0
        self.current_reward = 0
        self.limit_days = timedelta(days=limit_days)
        self.limit_candles_for_open_pos = 7 * 24

    def __call__(self, R):
        self.num_setups = 0
        self.current_reward = 0
        limit_datetime = self.dataset.get_base_datetime()

        self.model = self.model.to("cpu")
        self.model.set_value_only(True)
        self.model.eval()
        with torch.no_grad():
            for base_data_idx, data in enumerate(self.dataloader):
                _, _, _, _, base_data = data
                base_data = base_data[0]  # was a list of len BS, take the first element
                # check whether currently proposed datapoint is after the last setups end time
                if base_data[-1] <= limit_datetime:
                    continue

                print("Trying new")
                print(limit_datetime)
                # change labels for the model
                data[2][:, 0] = 1
                pos_out, _ = self.model(data)
                data[2][:, 0] = -1
                neg_out, _ = self.model(data)
                pos_out = pos_out.numpy()[0]  # get rid of BS dim and convert to np value
                neg_out = neg_out.numpy()[0]  # get rid of BS dim and convert to np value

                # convert back to the original price value
                pos_out = (pos_out * base_data[1]) + base_data[0]
                neg_out = (neg_out * base_data[1]) + base_data[0]

                # to see which prediction is reached first
                pos_time = self.dataset.time_price_reaches(start_time=base_data[-1], price=pos_out, direction=1)
                neg_time = self.dataset.time_price_reaches(start_time=base_data[-1], price=neg_out, direction=-1)

                if (pos_time is None) and (neg_time is None):
                    limit_datetime = base_data[-1]
                    continue
                elif pos_time is None:
                    # make pos time later than neg_time so that the setup opens for neg_time
                    pos_time = neg_time + timedelta(hours=1)
                elif neg_time is None:
                    # make neg_time later than pos_time so that the setup opens for pos_time
                    neg_time = pos_time + timedelta(hours=1)

                # if pos_time and neg_time are the same ignore this setup
                if pos_time == neg_time:
                    limit_datetime = base_data[-1]
                    continue

                entry_time = neg_time if neg_time < pos_time else pos_time
                # if the position opening time is a monday,
                # then don't open as we don't have monday data for the metadata encoder for model prediction
                # this is kind of a problem, but I cannot handle it for now
                if entry_time.dayofweek == 0:
                    print("Monday!")
                    limit_datetime = base_data[-1]
                    continue

                if entry_time - base_data[-1] > self.limit_days:
                    print("Long Shot:")
                    print(entry_time - base_data[-1])
                    limit_datetime = base_data[-1]
                    continue

                # returns the next datapoint equal to or just after the entry_time
                next_data = self._find_data_from_time(entry_time, base_data_idx)
                # probably it won't be None
                if next_data is None:
                    limit_datetime = base_data[-1]
                    continue

                # found the day to predict the next level
                # start building the setup
                # returns the next time to set as limit
                limit_datetime = self._create_and_evaluate_setup(pos_out, neg_out, pos_time, neg_time, next_data, R)

        self.model = self.model.to(self.orig_model_device)

        return self.current_reward, self.num_setups

    def _create_and_evaluate_setup(self, pos_out, neg_out, pos_time, neg_time, next_data, R):
        print("Found a setup!\n\n")

        side = 1 if neg_time < pos_time else -1
        entry_price = neg_out if neg_time < pos_time else pos_out
        entry_time = neg_time if neg_time < pos_time else pos_time
        _, _, _, _, next_base_data = next_data
        next_base_data = next_base_data[0]
        next_data[2][:, 0] = side
        tp_price, _ = self.model(next_data)
        tp_price = tp_price.numpy()[0]
        # convert back to the original price value
        tp_price = (tp_price * next_base_data[1]) + next_base_data[0]

        if side == 1:
            if tp_price < entry_price:
                print("Invalid TP price!")
                # self.model = self.model.to(self.orig_model_device)
                # assert False
                return entry_time
        if side == -1:
            if tp_price > entry_price:
                print("Invalid TP price!")
                # self.model = self.model.to(self.orig_model_device)
                # assert False
                return entry_time

        # max sl_distance is bounded by the range (or range / 2)
        sl_distance = min(abs((entry_price - tp_price) / R), next_base_data[1])
        sl_price = entry_price + (sl_distance * side * -1)
        cur_r = abs(entry_price - tp_price) / sl_distance

        tp_time = self.dataset.time_price_reaches(start_time=entry_time, price=tp_price, direction=side * 1)
        sl_time = self.dataset.time_price_reaches(start_time=entry_time, price=sl_price, direction=side * -1)
        limit_time_position, close_price = self.dataset.price_closes_after_n_candles(entry_time, self.limit_candles_for_open_pos)

        if limit_time_position is not None:
            self.num_setups += 1
            if (tp_time is not None) and (tp_time > limit_time_position):
                tp_time = None
            if (sl_time is not None) and (sl_time > limit_time_position):
                sl_time = None

            if (tp_time is not None) and (sl_time is not None):
                if tp_time < sl_time:
                    sl_time = None
                elif sl_time < tp_time:
                    tp_time = None
                # they are same, ignore
                else:
                    self.num_setups -= 1
                    limit_datetime = tp_time
                    return limit_datetime

            # both must be None or one must be None
            if tp_time is None and sl_time is None:
                cur_r = side * (close_price - entry_price) / sl_distance
                self.current_reward += cur_r
                limit_datetime = limit_time_position
                return limit_datetime

            # one must be None a
            elif tp_time is not None and sl_time is None:
                self.current_reward += cur_r
                limit_datetime = tp_time
                return limit_datetime
            elif sl_time is not None and tp_time is None:
                self.current_reward -= 1
                limit_datetime = sl_time
                return limit_datetime

            # won't ever reach here
            else:
                assert False

        # no need to build a setup anymore, not enough time left
        else:
            return self.dataset.get_limit_datetime()

    def _find_data_from_time(self, time, start_idx):
        # find the next setup data,
        while start_idx < len(self.dataloader):
            next_data = self.dataloader[start_idx]
            _, _, _, _, next_base_data = next_data
            next_base_data = next_base_data[0]
            if next_base_data[-1] >= time:
                return next_data
            start_idx += 1

        return None
