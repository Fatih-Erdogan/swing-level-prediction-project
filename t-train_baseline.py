import json
import os.path

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from utils import find_last_model, load_data, prepare_data
from dataloader.SwingDatasets import HourlyDataset
from dataloader.SuperTrainerDataset import SuperTrainerDataset
from trainer.trainer import HourlySwingModelTrainer
from model.baseline.baseline import LSTMBaseline

from data.dataset_paths import forex_path_dict, crypto_path_dict, index_path_dict, tz_dict

from metrics.trade_metric import TradeSetupMetric
from metrics.deviation_bins import DeviationEvaluator


if __name__ == "__main__":
    TRAIN = True
    EVALUATE_ON_METRICS = False

    pair_types_dict = {
        "forex": forex_path_dict,
        "crypto": crypto_path_dict,
        "index": index_path_dict
    }

    with open("baseline_config_train.json", "r") as file:
        config_train = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train cfg
    TYPE = config_train["type"]
    WINDOW_SIZE = config_train["window_size"]
    BATCH_SIZE = config_train["batch_size"]
    VALUE_ONLY = config_train["value_only"]
    LR = config_train["lr"]
    WD = config_train["wd"]
    OPTIMIZER = config_train["optimizer"]
    EPOCHS = config_train["epochs"]
    EVAL_PERIOD = config_train["eval_period"]
    CHECKPOINT_PERIOD = config_train["checkpoint_period"]
    MODEL_OUT_PATH = config_train["model_out"]

    # model cfg
    with open("baseline_config_model.json", "r") as file:
        config_model = json.load(file)

    inp_dim = config_model["inp_dim"]
    metadata_dim = config_model["metadata_dim"]
    metadata_bias = config_model["metadata_bias"]
    metadata_gate_bias = config_model["metadata_gate_bias"]
    hid_dim = config_model["hidden_dim"]
    lstm_num_layers = config_model["lstm_num_layers"]
    lstm_bidirectional = config_model["lstm_bidirectional"]
    lstm_dropout = config_model["lstm_dropout"]

    # load model
    model = LSTMBaseline(inp_dim=inp_dim, metadata_dim=metadata_dim, metadata_bias=metadata_bias,
                         metadata_gate_bias=metadata_gate_bias, hid_dim=hid_dim, num_layers=lstm_num_layers,
                         bidirectional=lstm_bidirectional, dropout=lstm_dropout)
    model.to(device)
    if config_train["load_model"] is not None:
        model_file = find_last_model(config_train["load_model"])
        print(f"Loading model from: {os.path.join(config_train['load_model'], model_file)}")
        model.load_state_dict(torch.load(os.path.join(config_train["load_model"], model_file), map_location=device))

    # pretrain in all pairs
    if config_train["pair"] == "all":
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
    # train for a single pair
    else:
        # load data
        pair = config_train["pair"]
        pair_data = load_data(pair_types_dict[TYPE][pair], add_zigzag_col=True)
        pair_data = prepare_data(pair_data, tz_dict[pair])
        train_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "train")
        dev_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "dev")
        test_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "test")

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=HourlyDataset.get_collate_fn())
        dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=HourlyDataset.get_collate_fn())
        # test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=HourlyDataset.get_collate_fn())

    if TRAIN:
        # save the cfg before start
        HourlySwingModelTrainer.save_cfg(config_model, config_train, out_path=MODEL_OUT_PATH)

        # train for value only
        if VALUE_ONLY:
            model.set_value_only(True)
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
            if OPTIMIZER == "adam":
                optimizer_cer = optim.Adam(model.parameters(), lr=LR)
            elif OPTIMIZER == "adamw":
                optimizer_cer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            else:
                assert False
            trainer = HourlySwingModelTrainer(model, train_dataloader, dev_dataloader, None,
                                              optimizer_cer, device)
            trainer.train(epochs=EPOCHS, eval_period=EVAL_PERIOD,
                          checkpoint_period=CHECKPOINT_PERIOD, out_path=os.path.join(MODEL_OUT_PATH, "cer"))

    if EVALUATE_ON_METRICS:
        assert config_train["pair"] != "all"
        print("Loading dataset...")
        metric_dataset = HourlyDataset(pair_data, WINDOW_SIZE, "dev")
        print("Evaluation starts...")

        bin_evaluator = DeviationEvaluator(model, metric_dataset, bin_size=0.1, limit_error=1.5)
        bin_evaluator.evaluate()
        bin_results = bin_evaluator.get_bin_dict(format="percentage")
        pretty_results = json.dumps(bin_results, indent=4)
        print(f"Total number of trials: {len(metric_dataset)}")
        print(pretty_results, end="\n\n")

        input("Press enter to run other metrics.")
        evaluator = TradeSetupMetric(model, metric_dataset, limit_days=7)
        r_1_results = evaluator(1)
        print(r_1_results)
        print("\n\n\n")
        input("Press enter to run other metrics.")
        r_2_results = evaluator(2)
        print(r_2_results)
        print("\n\n\n")
        input("Press enter to run other metrics.")
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
