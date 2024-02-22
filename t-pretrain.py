import json
import os.path

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from utils import find_last_model, load_data, prepare_data
from dataloader.SwingDatasets import HourlyDataset
from dataloader.SuperTrainerDataset import SuperTrainerDataset
from trainer.trainer import HourlySwingModelTrainer
from model.swing_model import HourlySwingModel

from data.dataset_paths import forex_path_dict, crypto_path_dict, index_path_dict, tz_dict

import argparse

if __name__ == "__main__":
    print("Pretraining begins...\n\n")
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--num', type=str, required=False, help='An integer number')
    args = parser.parse_args()
    config_no = args.num if args.num is not None else ""

    pair_types_dict = {
        "forex": forex_path_dict,
        "crypto": crypto_path_dict,
        "index": index_path_dict
    }

    with open(f"configs/config_pretrain{config_no}.json", "r") as file:
        config_train = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train cfg
    LOAD_MODEL = config_train["load_model"]
    TYPE = config_train["type"]
    WINDOW_SIZE = config_train["window_size"]
    BATCH_SIZE = config_train["batch_size"]
    VALUE_ONLY = config_train["value_only"]
    OPTIMIZER = config_train["optimizer"]
    LR = config_train["lr"]
    WD = config_train["wd"]
    EPOCHS = config_train["epochs"]
    EVAL_PERIOD = config_train["eval_period"]
    CHECKPOINT_PERIOD = config_train["checkpoint_period"]
    MODEL_OUT_PATH = config_train["model_out"]

    # model cfg
    with open(f"configs/config_model{config_no}.json", "r") as file:
        config_model = json.load(file)

    inp_dim = config_model["inp_dim"]
    metadata_dim = config_model["metadata_dim"]
    metadata_bias = config_model["metadata_bias"]
    metadata_gate_bias = config_model["metadata_gate_bias"]
    temporal_enricher_dropout = config_model["temporal_enricher_dropout"]
    fusion_model_dim = config_model["fusion_model_dim"]
    fusion_num_heads = config_model["fusion_num_heads"]
    fusion_num_layers = config_model["fusion_num_layers"]
    fusion_apply_grn = config_model["fusion_apply_grn"]
    fusion_dropout = config_model["fusion_dropout"]
    max_window = config_model["max_window"]
    positional_info = config_model["positional_info"]
    lstm_num_layers = config_model["lstm_num_layers"]
    lstm_bidirectional = config_model["lstm_bidirectional"]
    lstm_dropout = config_model["lstm_dropout"]
    loss_punish_cert = config_model["loss_punish_cert"]

    # load model
    model = HourlySwingModel(inp_dim=inp_dim, metadata_dim=metadata_dim, metadata_bias=metadata_bias,
                             metadata_gate_bias=metadata_gate_bias, 
                             temporal_enricher_dropout=temporal_enricher_dropout,
                             fusion_model_dim=fusion_model_dim,
                             fusion_num_heads=fusion_num_heads, fusion_num_layers=fusion_num_layers,
                             fusion_apply_grn=fusion_apply_grn, fusion_dropout=fusion_dropout,
                             max_window=max_window, positional_info=positional_info,
                             lstm_num_layers=lstm_num_layers, lstm_bidirectional=lstm_bidirectional,
                             lstm_dropout=lstm_dropout, loss_punish_cert=loss_punish_cert, device=device)
    model.to(device)

    if LOAD_MODEL is not None:
        model_file = find_last_model(LOAD_MODEL)
        print(f"Loading model from: {os.path.join(LOAD_MODEL, model_file)}")
        model.load_state_dict(torch.load(os.path.join(LOAD_MODEL, model_file), map_location=device))

    # load data and prepare super_dataset
    pair_path_dict = pair_types_dict[TYPE]
    super_train_dataset = SuperTrainerDataset()
    super_dev_dataset = SuperTrainerDataset()
    idx = 0
    for pair, path in pair_path_dict.items():
        idx += 1
        print(f"{idx} - Loading {pair.upper()} from {path}")
        # load data
        pair_data = load_data(path, add_zigzag_col=True)
        pair_data = prepare_data(pair_data, tz_dict[pair])
        # prepare dataset
        print(f"Building train dataset...")
        train_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "train")
        print(f"Building dev dataset...")
        dev_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "dev")
        # add to super dataset
        super_train_dataset.add_dataset(train_dataset)
        super_dev_dataset.add_dataset(dev_dataset)
        print(f"{pair.upper()} loaded.\n")

    train_dataloader = DataLoader(super_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=HourlyDataset.get_collate_fn())
    dev_dataloader = DataLoader(super_dev_dataset, batch_size=32, shuffle=False,
                                collate_fn=HourlyDataset.get_collate_fn())

    # save the cfg before start
    HourlySwingModelTrainer.save_cfg(config_model, config_train, out_path=MODEL_OUT_PATH)

    # train for value only
    if VALUE_ONLY:
        model.set_value_only(True)
        # optimizer_reg = optim.Adam(model.parameters(), lr=LR)
        if OPTIMIZER == "adam":
            optimizer_reg = optim.Adam(model.parameters(), lr=LR)
        elif OPTIMIZER == "adamw":
            optimizer_reg = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        else:
            assert False
        trainer = HourlySwingModelTrainer(model, train_dataloader, dev_dataloader, None,
                                          optimizer_reg, device)
        trainer.train(epochs=EPOCHS, eval_period=EVAL_PERIOD,
                      checkpoint_period=CHECKPOINT_PERIOD, out_path=os.path.join(MODEL_OUT_PATH, "reg"))

    # train for certitude as well
    else:
        model.set_value_only(False)
        optimizer_cer = optim.Adam(model.parameters(), lr=LR)
        trainer = HourlySwingModelTrainer(model, train_dataloader, dev_dataloader, None,
                                          optimizer_cer, device)
        trainer.train(epochs=EPOCHS, eval_period=EVAL_PERIOD,
                      checkpoint_period=CHECKPOINT_PERIOD, out_path=os.path.join(MODEL_OUT_PATH, "cer"))
