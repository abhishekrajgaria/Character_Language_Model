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


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")


def loadModel(model_instance, model_path):
    checkpoint = torch.load(model_path)
    # print(checkpoint)
    model_instance.load_state_dict(checkpoint["model_param"])
    print(
        f"""Dev_LAS of loaded model: {checkpoint["dev_metric"]} at epoch {checkpoint["epoch"]} with learning rate {checkpoint["learning_rate"]}"""
    )
    return model_instance


def evaluateModel(model, dataset):
    model = model.to(device)

    dataLoader = torch.utils.data.DataLoader(dataset, 128)
    with torch.no_grad():
        loss_function = nn.CrossEntropyLoss(
            weight=weights, ignore_index=ignore_index, reduce=False
        )

        loss_function = loss_function.to(device)
        model.eval()
        perplexity = 0

        total_len = 0
        for input, target in dataLoader:
            batch_size = input.shape[0]
            total_len += batch_size
            seq_len = input.shape[1]

            output = model(input.to(device))
            output = output.view(-1, output.shape[-1])
            target = target.reshape(-1)

            # seq_len x batch_size
            char_level_loss = loss_function(output, target.to(device))

            for i in range(batch_size):
                seq_loss = 0
                for j in range(i * seq_len, (i + 1) * seq_len):
                    seq_loss += char_level_loss[j]

                seq_loss = seq_loss / seq_len
                perplexity += 2**seq_loss

        perplexity = perplexity / total_len
        print(f"Perplexity on test dataset: {perplexity}")


if __name__ == "__main__":
    seq_len = 500

    train_folder = "./mp3_release/data/train"
    dev_folder = "./mp3_zrelease/data/dev"
    test_folder = "./mp3_release/data/test"

    vocab, rev_vocab = getVocab("./mp3_release/data/vocab.pkl")
    ignore_index = vocab["[PAD]"]

    train_char_data = getCharData(train_folder, vocab)
    # dev_char_data = getCharData(dev_folder, vocab)
    test_char_data = getCharData(test_folder, vocab)

    weights = getWeights(vocab, train_char_data)

    test_dataset = getData(vocab, test_char_data, 500)

    test_model_1 = model.LMModel(len(vocab), num_layers=1)
    best_model_path = (
        "/scratch/general/vast/u1471428/cs6957/assignment3/models/layer_1/0.01_0"
    )
    test_model_1 = loadModel(test_model_1, best_model_path)

    num_param_1 = sum(p.numel() for p in test_model_1.parameters())
    print(num_param_1)

    evaluateModel(test_model_1, test_dataset)

    test_model_2 = model.LMModel(len(vocab), num_layers=2)
    best_model_path = (
        "/scratch/general/vast/u1471428/cs6957/assignment3/models/layer_2/0.01_0"
    )
    test_model_2 = loadModel(test_model_2, best_model_path)

    num_param_2 = sum(p.numel() for p in test_model_2.parameters())
    print(num_param_2)

    evaluateModel(test_model_2, test_dataset)
