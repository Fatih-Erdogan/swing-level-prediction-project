import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import generate_vpvr
import torch


class HourlyDataset(Dataset):
    # metadata:
    # - vpvr (on prev week)

    # - prev weekly high
    # - prev weekly low
    # - prev weekly open
    # - prev weekly close
    # - monday open
    # - monday high
    # - monday low
    # - monday close

    # - bias vector
    # - 2 category (weekend, weekday{\Monday})
    def __init__(self, ltf_data, window_size, split=None):
        super(HourlyDataset, self).__init__()
        # at the end of init:
        # ltf_data:
        # ltf_candles:
        #    ['open', 'high', 'low', 'close', 'volume'] 'time' 'week_idx' 'usable' 'weekday'
        # weekly_data:
        #    ['pw_high', 'pw_low', 'pw_open', 'pw_close', 'mon_open', 'mon_high', 'mon_low', 'mon_close', 'vpvr']
        # usable_ltf_candles:
        #    ["week_idx", "weekday"]

        self.split = split
        self.ltf_data = ltf_data.copy()
        self.ltf_candles = self.ltf_data[["open", "high", "low", "close", "volume", "time"]].copy()
        self.labels = self.ltf_data[["label", "value"]].copy()

        self.ltf_candles = self._add_week_idx(self.ltf_candles)
        self.weekly_data = self._extract_weekly_info(self.ltf_candles)  # also adds "week_idx" to ltf_candles
        self.window_size = window_size

        self._handle_weekly_none()  # for weekly
        self._mark_usable_or_not()  # for ltf_candles
        self._guarantee_overlap_weekly_and_candles()  # for weekly and ltf_candles

        self._control_indices(self.ltf_candles)
        self.usable_ltf_candles = self.ltf_candles[self.ltf_candles.usable == 1][["week_idx", "weekday"]].copy()
        self.usable_ltf_candles = self._handle_consistent_split(self.usable_ltf_candles, self.split)  # handle split

        self._drop_unnecessary_and_order_ltf()  # for ltf_candles
        self._order_weekly_data()  # for weekly
        self.ltf_candles = self.ltf_candles.to_numpy()  # for efficiency

    def get_base_datetime(self):
        first_usable_idx = self.usable_ltf_candles.index[0]
        return self.ltf_data.loc[first_usable_idx].time

    def get_limit_datetime(self):
        last_usable_idx = self.usable_ltf_candles.index[-1]
        return self.ltf_data.loc[last_usable_idx].time

    def time_price_reaches(self, start_time, price, direction):
        # direction > 0 : or higher
        # direction < 0 : or lower
        if direction > 0:
            subset = self.ltf_data[(self.ltf_data['time'] > start_time) &
                                   (self.ltf_data['high'] >= price)]
            return subset.iloc[0].time if not subset.empty else None
        if direction < 0:
            subset = self.ltf_data[(self.ltf_data['time'] > start_time) &
                                   (self.ltf_data['low'] <= price)]
            return subset.iloc[0].time if not subset.empty else None
        else:
            subset = self.ltf_data[(self.ltf_data['time'] > start_time) &
                                   ((self.ltf_data['low'] <= price) &
                                    (self.ltf_data['high'] >= price))]
            assert False
            # return subset.iloc[0].time if not subset.empty else None

    def price_closes_after_n_candles(self, base_time, num_candles):
        the_candle_candidates = self.ltf_data[self.ltf_data["time"] > base_time]
        if len(the_candle_candidates) > num_candles + 1:
            the_candle = the_candle_candidates.iloc[num_candles]
            return the_candle["time"], the_candle["close"]

        return None, None

    def __len__(self):
        return len(self.usable_ltf_candles)

    def __getitem__(self, idx):
        """
        :return:
        x            -> numpy array: window_size * OHLCV
        metadata[2:] -> numpy array: ['pw_open', 'pw_close', 'mon_open', 'mon_high', 'mon_low', 'mon_close', 'vpvr']
        label        -> numpy array: [direction, value]
        weekday      -> int: 1 if weekday, 0 if weekend
        """
        # return candle_seq, metadata (about the week), labels, weekday info
        #
        last_candle = self.usable_ltf_candles.iloc[idx]
        weekday = last_candle.weekday
        idx_in_df = last_candle.name
        last_candle_datetime = self.ltf_data.loc[idx_in_df].time

        # x -> row_idx * OHLCV
        x = self.ltf_candles[idx_in_df - (self.window_size - 1): idx_in_df + 1].copy()
        weekly_info = self.weekly_data.loc[last_candle.week_idx].to_numpy().copy()
        label = self.labels.loc[idx_in_df].to_numpy().copy()

        # normalize x and metadata components
        x, weekly_info, label, base_info = self._normalize_data_metadata_label(x, weekly_info, label)
        base_info.append(last_candle_datetime)

        return x, weekly_info[2:], label, weekday, base_info

    def _normalize_data_metadata_label(self, x, metadata, label):
        # x -> row * OHLCV
        # metadata -> series obj with
        #     ['pw_high', 'pw_low', 'pw_open', 'pw_close', 'mon_open', 'mon_high', 'mon_low', 'mon_close', 'vpvr']
        # label -> label, value

        # take the pw_high and pw_low -> index:
        #           OHLC of x;
        #           'pw_open', 'pw_close', 'mon_open', 'mon_high', 'mon_low', 'mon_close' of metadata;
        #           value of label.
        pw_high = metadata[0]
        pw_low = metadata[1]
        pw_mid = (pw_low + pw_high) / 2
        pw_half_range = (pw_high - pw_low) / 2

        x[:, :4] = (x[:, :4] - pw_mid) / pw_half_range
        metadata[2: 8] = (metadata[2: 8] - pw_mid) / pw_half_range
        label[1] = (label[1] - pw_mid) / pw_half_range

        # standardize the volume
        mean_vol = np.mean(x[:, -1])
        std_vol = np.std(x[:, -1])
        x[:, -1] = (x[:, -1] - mean_vol) / std_vol

        return x, metadata, label, [pw_mid, pw_half_range]

    def _handle_consistent_split(self, data, split):
        # split the data by separating based on the previous week
        # 0.7 - 0.2 - 0.1 for train - dev - test
        if split is None:
            return data

        train_prop = 0.7
        dev_prop = 0.2
        test_prop = 0.1

        num_rows = len(data)
        train_rows = int(num_rows * train_prop)
        dev_rows = int(num_rows * dev_prop)

        # Calculate the indices for splitting
        train_split_index = train_rows
        dev_split_index = train_rows + dev_rows

        # Split the DataFrame
        if split == "train":
            train_data = data.iloc[:train_split_index].copy()
            return train_data
        elif split == "dev":
            dev_data = data.iloc[train_split_index:dev_split_index].copy()
            return dev_data
        elif split == "test":
            test_data = data.iloc[dev_split_index:].copy()
            return test_data

        assert False, "Split should be 'train', 'dev', or 'test'"

    """
    def _handle_consistent_split(self, data, split):
        # split the data by separating based on the previous week
        # 0.7 - 0.2 - 0.1 for train - dev - test
        if split is None:
            return data

        random_state = 42
        train_prop = 0.7
        dev_prop = 0.2
        test_prop = 0.1

        dev_remain = dev_prop / (test_prop + dev_prop)
        test_remain = test_prop / (test_prop + dev_prop)

        unique_week_indices = pd.Series(data.week_idx.unique())
        train_weeks = unique_week_indices.sample(frac=train_prop, random_state=random_state)
        if split == "train":
            return data[data['week_idx'].isin(train_weeks)].copy()

        unique_week_indices = unique_week_indices[~unique_week_indices.isin(train_weeks)]
        dev_weeks = unique_week_indices.sample(frac=dev_remain, random_state=random_state)
        if split == "dev":
            return data[data['week_idx'].isin(dev_weeks)].copy()

        unique_week_indices = unique_week_indices[~unique_week_indices.isin(dev_weeks)]
        test_weeks = unique_week_indices
        if split == "test":
            return data[data['week_idx'].isin(test_weeks)].copy()

        assert False, "Split should be 'train', 'dev', or 'test'"


        def _handle_consistent_split(self, data, split):
        # 0.7 - 0.2 - 0.1 for train - dev - test
        if split is None:
            return data

        random_state = 42
        train_prop = 0.7
        dev_prop = 0.2
        test_prop = 0.1

        dev_remain = dev_prop / (test_prop + dev_prop)
        test_remain = test_prop / (test_prop + dev_prop)

        train_split = data.sample(frac=train_prop, random_state=random_state)
        if split == "train":
            return train_split

        train_idx = train_split.index
        dev_test_data = data.drop(train_idx, axis=0)
        dev_split = dev_test_data.sample(frac=dev_remain, random_state=random_state)
        if split == "dev":
            return dev_split

        dev_idx = dev_split.index
        test_split = dev_test_data.drop(dev_idx, axis=0)
        if split == "test":
            return test_split

        assert False, "Split should be 'train', 'dev', or 'test'"
    """

    def _order_weekly_data(self):
        order = ['pw_high', 'pw_low', 'pw_open', 'pw_close', 'mon_open', 'mon_high', 'mon_low', 'mon_close', 'vpvr']
        self.weekly_data = self.weekly_data[order]

    def _drop_unnecessary_and_order_ltf(self):
        keep_id = ["open", "high", "low", "close", "volume"]
        for col in self.ltf_candles.columns:
            if col not in keep_id:
                self.ltf_candles.drop(col, inplace=True, axis=1)

        # also order the columns
        self.ltf_candles = self.ltf_candles[keep_id]

    def _control_indices(self, df):
        is_ordered = df.index.equals(pd.RangeIndex(start=0, stop=len(df)))
        assert is_ordered, "The indices must not be corrupted!"

    def _handle_weekly_none(self):
        while self.weekly_data.iloc[0].isna().any():
            self.weekly_data = self.weekly_data.iloc[1:]

        while self.weekly_data.iloc[-1].isna().any():
            self.weekly_data = self.weekly_data.iloc[:-1]

        if len(self.weekly_data[self.weekly_data.isna().any(axis=1)]) > 0:
            none_rows = self.weekly_data[self.weekly_data.isna().any(axis=1)]
            print(self.weekly_data[self.weekly_data.isna().any(axis=1)])
            assert False, "Weekly data contains None values!"

    def _guarantee_overlap_weekly_and_candles(self):
        # assumes no None value in weekly data
        idx_array = self.ltf_candles[self.ltf_candles.usable == 1].week_idx.unique()
        weekly_idx_array = self.weekly_data.index.to_numpy()
        for idx in idx_array:
            if idx not in weekly_idx_array:
                print(idx)
                assert False, "Found a non-existing week in weekly data"

    def _mark_usable_or_not(self):
        # policy of not usable:
        # if monday or among first (window_size - 1) candles, then not usable
        # also marks whether weekday or weekend
        self.ltf_candles["usable"] = self.ltf_candles.apply \
            (lambda row: 0 if (row.name < (self.window_size - 1) or row["time"].weekday() == 0) else 1, axis=1)
        self.ltf_candles["weekday"] = self.ltf_candles.apply(lambda row: 0 if row["time"].weekday() > 4 else 1, axis=1)

    def _add_week_idx(self, ltf_data):
        # already call by reference but still return it
        # add the week_idx column to the original data
        # indices are monday date
        ltf_data["week_idx"] = (
                ltf_data.time - pd.to_timedelta(ltf_data.time.dt.weekday, unit='D')).dt.strftime(
            "%Y-%m-%d")
        return ltf_data

    def _extract_weekly_info(self, hourly_candles):
        time_idx_candles = hourly_candles.set_index("time", inplace=False)

        # Monday ohlc
        monday_data = time_idx_candles[time_idx_candles.index.weekday == 0]
        monday_ohlc = monday_data.resample("W-MON").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        monday_ohlc.index = monday_ohlc.index.strftime('%Y-%m-%d')
        monday_ohlc.columns = ["mon_open", "mon_high", "mon_low", "mon_close"]

        # handle the case monday data was null
        weekday_idx = 1
        map_day_str_dict = {1: "W-TUE", 2: "W-WED", 3: "W-THU"}
        while len(monday_ohlc[monday_ohlc.isna().any(axis=1)]) > 0:
            if weekday_idx == 4:
                assert False, "Problem in Monday data"

            print(f"Trying to replace {len(monday_ohlc[monday_ohlc.isna().any(axis=1)])} " +
                  f"monday data (0) with ({weekday_idx})")
            day_data = time_idx_candles[time_idx_candles.index.weekday == weekday_idx]
            day_ohlc = day_data.resample(map_day_str_dict[weekday_idx]).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })
            day_ohlc.index = day_ohlc.index - pd.Timedelta(days=weekday_idx)
            day_ohlc.index = day_ohlc.index.strftime('%Y-%m-%d')
            day_ohlc.columns = ["mon_open", "mon_high", "mon_low", "mon_close"]

            null_indices = monday_ohlc[monday_ohlc.isnull().any(axis=1)].index
            monday_ohlc.loc[null_indices] = day_ohlc.loc[null_indices]

            weekday_idx += 1

        # weekly ohlc
        weekly_data = time_idx_candles.resample("W-SUN").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"})

        # make index monday string
        # week indices were already the final day being Sunday
        # now we push them by 1 day making it following Monday
        weekly_data.index = weekly_data.index + pd.Timedelta(days=1)
        assert all(weekly_data.index.dayofweek == 0), "All dates in weekly_data need to refer to a Monday"
        weekly_data.index = weekly_data.index.strftime("%Y-%m-%d")
        weekly_data.columns = ["pw_open", "pw_high", "pw_low", "pw_close"]

        if (len(weekly_data) - len(monday_ohlc)) > 1:
            assert False

        weekly_data = pd.merge(weekly_data, monday_ohlc, left_index=True, right_index=True, how='outer')

        # VPVR
        vpvr_series = pd.Series(dtype=object)
        num_bins = 10
        for week_idx in time_idx_candles['week_idx'].unique():
            # current week
            weekly_box = time_idx_candles[time_idx_candles['week_idx'] == week_idx]

            if not weekly_box.empty:
                buy_array, sell_array = generate_vpvr(weekly_box, num_bins)
                # concatenate buy and sell arrays
                vpvr_series.at[week_idx] = np.concatenate([buy_array, sell_array])

        return pd.merge(weekly_data, vpvr_series.rename("vpvr"), left_index=True, right_index=True, how='outer')

    @staticmethod
    def get_collate_fn():
        def hourly_collate_fn(batch):
            x = [item[0] for item in batch]
            metadata = [item[1] for item in batch]
            label = [item[2] for item in batch]
            weekday = [item[3] for item in batch]
            base_data = [item[4] for item in batch]

            for idx, met in enumerate(metadata):
                metadata[idx] = np.concatenate((met[0:-1], met[-1]))

            # Convert lists to PyTorch tensors
            x_tensor = torch.tensor(x, dtype=torch.float32)  # shape: [BS, seq_len, OHLCV]
            metadata_tensor = torch.tensor(metadata, dtype=torch.float32)  # shape: [BS, metadata_dim]
            label_tensor = torch.tensor(label, dtype=torch.float32)  # shape: [BS, 2]
            weekday_tensor = torch.tensor(weekday, dtype=torch.float32)  # shape: [BS]

            return x_tensor, metadata_tensor, label_tensor, weekday_tensor, base_data

        return hourly_collate_fn
