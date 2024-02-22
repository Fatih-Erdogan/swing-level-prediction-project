import torch
import os
import json
from tqdm import tqdm


class HourlySwingModelTrainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, optimizer, device):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.train_losses = list()
        self.val_losses = list()

    def train(self, epochs, eval_period, checkpoint_period, out_path, eval_first=False):
        if eval_first:
            self.evaluate()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as t:
                batch_count = 0
                # start_time = time.time()
                for inputs in t:
                    inputs = [item.to(self.device) if i < len(inputs) - 1 else item for i, item in enumerate(inputs)]
                    batch_count += 1
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = outputs[-1]
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    t.set_postfix(loss=total_loss / batch_count)
                # end_time = time.time()
                # print(f"Validation completed in {end_time - start_time} seconds.")
                # print(f"Current loss: {total_loss / batch_count}")

            self.train_losses.append(total_loss / batch_count)

            if (epoch + 1) % eval_period == 0:
                self.evaluate()

            if (epoch + 1) % checkpoint_period == 0:
                self.save_checkpoint(out_path, epoch)
                self.save_losses(out_path)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with tqdm(self.dev_loader, desc="Evaluating") as t:
                batch_count = 0
                # start_time = time.time()
                for inputs in t:
                    batch_count += 1
                    inputs = [item.to(self.device) if i < len(inputs) - 1 else item for i, item in enumerate(inputs)]
                    outputs = self.model(inputs)
                    loss = outputs[-1]
                    total_loss += loss.item()
                    t.set_postfix(loss=total_loss / batch_count)
                # end_time = time.time()
                # print(f"Validation completed in {end_time - start_time} seconds.")
                # print(f"Validation loss: {total_loss / batch_count}")

            self.val_losses.append(total_loss / batch_count)

    def save_checkpoint(self, out_path, epoch):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        torch.save(self.model.state_dict(), f"{out_path}/model_epoch_{epoch+1}.pth")

    def save_losses(self, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        loss_results_path = os.path.join(out_path, "loss_results.txt")
        with open(loss_results_path, "w") as file:
            file.write("Training Losses:\n")
            for loss in self.train_losses:
                file.write(f"{loss}\n")
            file.write("\n")
            file.write("Validation Losses:\n")
            for loss in self.val_losses:
                file.write(f"{loss}\n")

        print(f"Losses saved to {loss_results_path}")

    @staticmethod
    def save_cfg(model_cfg, train_cfg, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        model_cfg_path = os.path.join(out_path, "model_config.json")
        with open(model_cfg_path, "w") as file:
            json.dump(model_cfg, file, indent=4)
        print(f"Model configuration saved to {model_cfg_path}")

        train_cfg_path = os.path.join(out_path, "training_config.json")
        with open(train_cfg_path, "w") as file:
            json.dump(train_cfg, file, indent=4)
        print(f"Training configuration saved to {train_cfg_path}")



