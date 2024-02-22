import pandas as pd
import numpy as np
import os
import re


def find_last_model(path):
    model_files = [f for f in os.listdir(path) if f.endswith('.pth')]
    max_epoch = -1
    model_to_load = ""
    for file in model_files:
        match = re.search(r"model_epoch_(\d+).pth", file)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                model_to_load = file

    return model_to_load


def prepare_data(data_df, tz):
    handle_zigzag(data_df)
    data_df.drop(["Volume MA", "RelativeIndice", "HighOrLow"], axis=1, inplace=True)
    data_df = data_df[['time', 'open', 'high', 'low', 'close', 'Volume', 'zigzag']].copy()
    data_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'zigzag']
    data_df['time'] = pd.to_datetime(data_df['time'], utc=True)
    data_df['time'] = data_df['time'].dt.tz_convert(tz)
    label_data(data_df)
    data_df = data_df[data_df.value.notna()]
    assert data_df.shape[0] == (data_df.index[-1] + 1)
    assert len(data_df[data_df.isna().any(axis=1)]) == 0

    return data_df


def load_data(file_path, add_zigzag_col=True):
    data = pd.read_csv(file_path)
    if add_zigzag_col:
        data["zigzag"] = 0
    return data


def handle_zigzag(df):
    df_indices = df[df.RelativeIndice != 0]

    base_indices = df_indices.index.to_numpy()
    relative_indices = df_indices.RelativeIndice.to_numpy()
    zigzag_values = df_indices.HighOrLow.to_numpy()

    target_indices = base_indices - relative_indices
    mask = target_indices >= 0

    target_indices = target_indices[mask]
    zigzag_values = zigzag_values[mask]
    assert len(target_indices) == len(zigzag_values)

    df.loc[target_indices, "zigzag"] = zigzag_values


def label_data(df):
    labeling_data = df[df.zigzag != 0]
    limit_indices = labeling_data.index.to_numpy()
    core_labels = labeling_data.zigzag.to_numpy()
    core_values = np.select([labeling_data['zigzag'] == 1, labeling_data['zigzag'] == -1],
                            [labeling_data["high"], labeling_data["low"]],
                            default=np.nan)

    values_list = list()
    labels_list = list()
    limit_indices = np.append(np.array([0]), limit_indices)
    for i in range(len(limit_indices) - 1):
        low_idx = limit_indices[i]
        high_idx = limit_indices[i + 1]
        values_list.extend([core_values[i]] * (high_idx - low_idx))
        labels_list.extend([core_labels[i]] * (high_idx - low_idx))

    values_list = np.array(values_list)
    values_list = np.append(values_list, np.full(len(df) - len(labels_list), np.nan))
    labels_list = np.array(labels_list)
    labels_list = np.append(labels_list, np.full(len(df) - len(labels_list), np.nan))

    df["label"] = labels_list
    df["value"] = values_list

    # handle if there exist a sequence of same label with different target value
    # it should be same for a given seq
    # that could be caused by the data itself from tradingview
    zigzag_idx = df[df.zigzag != 0].index
    base_idx = 0
    for idx in zigzag_idx:
        label = df.loc[base_idx].label
        func = max if label == 1 else min
        new_val = func(df.loc[base_idx:idx - 1, "value"])
        df.loc[base_idx:idx - 1, "value"] = new_val
        base_idx = idx


def generate_vpvr(df, num_of_bins):
    """
    Calculates VPVR on the df

    """
    # last_bars_data = df.iloc[-num_of_bars:]
    last_bars_data = df
    assert len(last_bars_data) <= 168
    range_high = last_bars_data['high'].max()
    range_low = last_bars_data['low'].min()
    range_height = range_high - range_low

    low_list = [range_low + range_height * i / num_of_bins for i in range(num_of_bins)]
    high_list = [range_low + range_height * (i + 1) / num_of_bins for i in range(num_of_bins)]
    mid_list = [(low + high) / 2 for low, high in zip(low_list, high_list)]

    buy_volume_list = np.zeros(num_of_bins)
    sell_volume_list = np.zeros(num_of_bins)

    for index, row in last_bars_data.iterrows():
        current_bar_height = row['high'] - row['low']
        current_buy_volume = 0 if current_bar_height == 0 else row['volume'] * (
                    row['close'] - row['low']) / current_bar_height
        current_sell_volume = 0 if current_bar_height == 0 else row['volume'] * (
                    row['high'] - row['close']) / current_bar_height

        for j in range(num_of_bins):
            histogram_low = low_list[j]
            histogram_high = high_list[j]
            target = max(histogram_high, row['high']) - min(histogram_low, row['low']) \
                     - (max(histogram_high, row['high']) - min(histogram_high, row['high'])) \
                     - (max(histogram_low, row['low']) - min(histogram_low, row['low']))
            histogram_volume_percentage = target / current_bar_height if current_bar_height != 0 else 0

            if histogram_volume_percentage > 0:
                buy_volume_list[j] += current_buy_volume * histogram_volume_percentage
                sell_volume_list[j] += current_sell_volume * histogram_volume_percentage

    # scale the values between 0 - 1
    total_vol = (buy_volume_list.sum() + sell_volume_list.sum())
    buy_volume_list = buy_volume_list / total_vol
    sell_volume_list = sell_volume_list / total_vol

    return buy_volume_list, sell_volume_list

