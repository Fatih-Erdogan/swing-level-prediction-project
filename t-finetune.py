import json
import os.path

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from utils import find_last_model, load_data, prepare_data
from dataloader.SwingDatasets import HourlyDataset
from trainer.trainer import HourlySwingModelTrainer
from model.swing_model import HourlySwingModel

from data.dataset_paths import forex_path_dict, crypto_path_dict, index_path_dict, tz_dict

from metrics.trade_metric import TradeSetupMetric
from metrics.deviation_bins import DeviationEvaluator

import argparse

if __name__ == "__main__":
    print("Finetuning begins...\n\n")
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("-t", action="store_true", help="Set this flag to train/finetune the model")
    parser.add_argument("-e", action="store_true", help="Set this flag to evaluate the model")
    args = parser.parse_args()
    TRAIN = args.t
    EVALUATE_ON_METRICS = args.e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_types_dict = {
        "forex": forex_path_dict,
        "crypto": crypto_path_dict,
        "index": index_path_dict
    }

    with open("configs/finetune_train_config.json", "r") as file:
        config_train = json.load(file)

    # train cfg
    TYPE = config_train["type"]
    pair_path_dict = pair_types_dict[TYPE]
    LOAD_MODEL = config_train["load_model"]
    PAIR_PATH = pair_path_dict[config_train["pair"]]
    PAIR_TZ = tz_dict[config_train["pair"]]
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
    with open("configs/finetune_model_config.json", "r") as file:
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
                             metadata_gate_bias=metadata_gate_bias, fusion_model_dim=fusion_model_dim,
                             temporal_enricher_dropout=temporal_enricher_dropout,
                             fusion_num_heads=fusion_num_heads, fusion_num_layers=fusion_num_layers,
                             fusion_apply_grn=fusion_apply_grn, fusion_dropout=fusion_dropout,
                             max_window=max_window, positional_info=positional_info,
                             lstm_num_layers=lstm_num_layers, lstm_bidirectional=lstm_bidirectional,
                             lstm_dropout=lstm_dropout, loss_punish_cert=loss_punish_cert, device=device)
    model.to(device)

    if LOAD_MODEL is not None and TRAIN:
        if LOAD_MODEL[-3:] == "pth":
            print(f"Loading model from: {LOAD_MODEL}")
            model.load_state_dict(torch.load(LOAD_MODEL, map_location=device))
        else:
            model_file = find_last_model(LOAD_MODEL)
            print(f"Loading model from: {os.path.join(LOAD_MODEL, model_file)}")
            model.load_state_dict(torch.load(os.path.join(LOAD_MODEL, model_file), map_location=device))

    elif not TRAIN and EVALUATE_ON_METRICS:
        model_path = LOAD_MODEL
        # model_file = find_last_model(model_path)
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # load data
    pair_data = load_data(PAIR_PATH, add_zigzag_col=True)
    pair_data = prepare_data(pair_data, PAIR_TZ)

    if TRAIN:
        train_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "train")
        dev_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "dev")
        test_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "test")

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=HourlyDataset.get_collate_fn())
        dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False,
                                    collate_fn=HourlyDataset.get_collate_fn())
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                                     collate_fn=HourlyDataset.get_collate_fn())

        print(f"Started training for {config_train['pair']} ...")
        # save the cfg before start
        HourlySwingModelTrainer.save_cfg(config_model, config_train, out_path=MODEL_OUT_PATH)

        # train for value only
        if VALUE_ONLY:
            model.set_value_only(True)
            if OPTIMIZER == "adam" or OPTIMIZER is None:
                optimizer_reg = optim.Adam(model.parameters(), lr=LR)
            elif OPTIMIZER == "adamw":
                optimizer_reg = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            else:
                assert False
            trainer = HourlySwingModelTrainer(model, train_dataloader, dev_dataloader, test_dataloader,
                                              optimizer_reg, device)
            trainer.train(epochs=EPOCHS, eval_period=EVAL_PERIOD,
                          checkpoint_period=CHECKPOINT_PERIOD, out_path=os.path.join(MODEL_OUT_PATH, "reg"),
                          eval_first=True)

        # train for certitude as well
        else:
            model.set_value_only(False)
            if OPTIMIZER == "adam" or OPTIMIZER is None:
                optimizer_cer = optim.Adam(model.parameters(), lr=LR)
            elif OPTIMIZER == "adamw":
                optimizer_cer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            else:
                assert False
            trainer = HourlySwingModelTrainer(model, train_dataloader, dev_dataloader, test_dataloader,
                                              optimizer_cer, device)
            trainer.train(epochs=EPOCHS, eval_period=EVAL_PERIOD,
                          checkpoint_period=CHECKPOINT_PERIOD, out_path=os.path.join(MODEL_OUT_PATH, "cer"),
                          eval_first=True)

    if EVALUATE_ON_METRICS:
        print("Loading dataset...")
        metric_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "test")
        print("Evaluation starts...")

        bin_evaluator = DeviationEvaluator(model, metric_dataset, bin_size=0.1, limit_error=1.5)
        bin_evaluator.evaluate()
        bin_results = bin_evaluator.get_bin_dict(format="percentage")
        pretty_results = json.dumps(bin_results, indent=4)
        print(f"Total number of trials: {len(metric_dataset)}")
        print(pretty_results, end="\n\n")

        evaluator = TradeSetupMetric(model, metric_dataset, limit_days=7)
        r_1_results = evaluator(1)
        print(r_1_results)
        print("\n\n\n")

        r_2_results = evaluator(2)
        print(r_2_results)
        print("\n\n\n")

        r_3_results = evaluator(3)
        print(r_3_results)

        # Save results
        file_path = os.path.join(MODEL_OUT_PATH, "metric_results.txt")
        with open(file_path, 'w') as file:
            file.write("Bin Distribution (Indexed at half range):\n")
            file.write(pretty_results + "\n\n")

            file.write("1R Results:\n")
            file.write(json.dumps(r_1_results) + "\n\n")

            file.write("2R Results:\n")
            file.write(json.dumps(r_2_results) + "\n\n")

            file.write("3R Results:\n")
            file.write(json.dumps(r_3_results) + "\n\n")
