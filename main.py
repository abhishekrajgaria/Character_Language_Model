import torch
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import model

from custom_data_loader import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, help="Directory where model checkpoints will be saved"
)
args = parser.parse_args()
output_dir = args.output_dir

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")

best_perf_dict = {
    "metric": -1,
    "model_param": None,
    "optim_param": None,
    "epoch": 0,
    "learning_rate": 0,
}


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        dev_dataset,
        weights,
        ignore_index,
        batch_size,
        epochs,
        num_layer,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = weights
        self.ignore_index = ignore_index
        self.num_layer = num_layer

    def trainModel(self, learning_rate):
        loss_function = nn.CrossEntropyLoss(
            weight=self.weights, ignore_index=self.ignore_index
        )

        self.model = self.model.to(device)
        loss_function = loss_function.to(device)
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

        train_dataLoader = torch.utils.data.DataLoader(
            self.train_dataset, self.batch_size, shuffle=True
        )
        dev_dataLoader = torch.utils.data.DataLoader(
            self.dev_dataset, self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            train_loss = []

            for input, target in train_dataLoader:
                self.model.train()
                optimizer.zero_grad()
                output = self.model(
                    input.to(device)
                )  # sen_len * batch_size * vocab_size
                # print("output ", output.shape)
                # target = target.permute(1, 0)
                # print("target", target.shape)

                # (batch_size x seq_len, vocab_size)
                output = output.view(-1, output.shape[-1])

                # batch_size x seq_len
                target = target.reshape(-1)

                loss = loss_function(output, target.to(device))

                # print(loss.shape)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.cpu().item())

            print(f"Average training batch loss: {np.mean(train_loss)}")

            # here dev loop
            with torch.no_grad():
                dev_loss_function = nn.CrossEntropyLoss(
                    weight=self.weights, ignore_index=self.ignore_index, reduce=False
                )

                dev_loss_function = dev_loss_function.to(device)
                self.model.eval()
                dev_perplexity = 0

                dev_len = 0

                for input, target in dev_dataLoader:
                    batch_size = input.shape[0]
                    dev_len += batch_size
                    seq_len = input.shape[1]

                    output = self.model(input.to(device))
                    # sen_len * batch_size * vocab_size
                    # print("output ", output.shape)
                    # target = target.permute(1, 0)
                    # print("target", target.shape)

                    output = output.view(-1, output.shape[-1])
                    target = target.reshape(-1)

                    char_level_loss = dev_loss_function(
                        output, target.to(device)
                    )  # seq_len x batch_size

                    # print(char_level_loss.shape)

                    for i in range(batch_size):
                        seq_loss = 0
                        for j in range(i * seq_len, (i + 1) * seq_len):
                            seq_loss += char_level_loss[j]

                        seq_loss = seq_loss / seq_len
                        dev_perplexity += 2**seq_loss
                    del char_level_loss

                dev_perplexity = dev_perplexity / dev_len

                print(f"Perplexity on dev dataset: {dev_perplexity}")

            if (
                best_perf_dict["metric"] == -1
                or dev_perplexity < best_perf_dict["metric"]
            ):
                # print("hello")
                # best_perf_dict["model_param"]: self.model.state_dict()
                # best_perf_dict["optim_param"]: optimizer.state_dict()
                best_perf_dict["metric"] = dev_perplexity
                best_perf_dict["epoch"] = epoch
                best_perf_dict["learning_rate"] = learning_rate

            torch.save(
                {
                    "model_param": self.model.state_dict(),
                    "optim_param": optimizer.state_dict(),
                    "dev_metric": dev_perplexity,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                },
                f"{output_dir}/layer_{self.num_layer}/{learning_rate}_{epoch}",
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    learning_rates = [0.0001, 0.00001, 0.000001]

    seq_len = 500

    train_folder = "./mp3_release/data/train"
    dev_folder = "./mp3_release/data/dev"
    test_folder = "./mp3_release/data/test"
    vocab, _ = getVocab("./mp3_release/data/vocab.pkl")

    ignore_index = vocab["[PAD]"]

    train_char_data = getCharData(train_folder, vocab)
    dev_char_data = getCharData(dev_folder, vocab)
    test_char_data = getCharData(test_folder, vocab)

    weights = getWeights(vocab, train_char_data)

    train_dataset = getData(vocab, train_char_data, 500)
    dev_dataset = getData(vocab, dev_char_data, 500)
    test_dataset = getData(vocab, test_char_data, 500)

    model1 = model.LMModel(len(vocab))

    start_time = time.time()

    for lr in learning_rates:
        print("*****************")
        print("learning_rate", lr)
        trainer = Trainer(
            model1,
            train_dataset,
            dev_dataset,
            weights,
            ignore_index,
            batch_size=128,
            epochs=5,
            num_layer=1,
        )
        trainer.trainModel(lr)
        print("*****************")
        print()

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

    model2 = model.LMModel(len(vocab), num_layers=2)

    start_time = time.time()

    for lr in learning_rates:
        print("*****************")
        print("learning_rate", lr)
        trainer = Trainer(
            model2,
            train_dataset,
            dev_dataset,
            weights,
            ignore_index,
            batch_size=128,
            epochs=5,
            num_layer=2,
        )
        trainer.trainModel(lr)
        print("*****************")
        print()

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
