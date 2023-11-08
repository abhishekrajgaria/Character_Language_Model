import time
import torch
import numpy as np
from mp3_release.scripts import utils

import pickle


def getVocab(filename):
    with open(filename, "rb") as f:
        vocab2idx = pickle.load(f)
        idx2vocab = {}
        for key in vocab2idx.keys():
            idx2vocab[vocab2idx[key]] = key
        return vocab2idx, idx2vocab


def getCharData(filename, vocab):
    files = utils.get_files(filename)
    data = utils.convert_files2idx(files, vocab)
    return data


def getWeights(vocab, data):
    chars = vocab.keys()
    dict = {}
    for char in chars:
        dict[char] = 0

    total_cnt = 0
    for line in data:
        for char in line:
            if char in chars:
                dict[char] += 1
            else:
                dict["<unk>"] += 1
            total_cnt += 1

    weights = [0 for i in range(len(chars))]
    for char in dict.keys():
        weights[vocab[char]] = 1 - (dict[char] / total_cnt)

    return torch.tensor(weights)


def convertData2Seq(seq_len, data, vocab, limit):
    input_data = []
    target_data = []
    for line in data:  # [:limit]:
        line_size = len(line)
        # print(line_size)
        start = 0
        while (line_size - start) > seq_len:
            input_data.append(line[start : start + seq_len])
            target_data.append(line[start + 1 : start + seq_len + 1])
            start += seq_len
        temp_input = line[start:]
        diff = seq_len - len(temp_input)
        temp_input = temp_input + [vocab["[PAD]"]] * diff

        temp_target = line[start + 1 :]
        temp_target = temp_target + [vocab["[PAD]"]] * (diff + 1)
        input_data.append(temp_input)
        target_data.append(temp_target)

    return input_data, target_data


def convertData2Tensor(data):
    tensor_data = torch.tensor(data)
    print(tensor_data.shape)
    return tensor_data


def getData(vocab, data, seq_len):
    # data = getCharData(data_folder, vocab)
    print(len(data))
    input_seq_data, target_seq_data = convertData2Seq(seq_len, data, vocab, 3000)

    input_tensor_data = convertData2Tensor(input_seq_data)
    target_tensor_data = convertData2Tensor(target_seq_data)

    dataset = torch.utils.data.TensorDataset(input_tensor_data, target_tensor_data)
    return dataset

    # return tensor_data


if __name__ == "__main__":
    start_time = time.time()
    vocab, _ = getVocab("./mp3_release/data/vocab.pkl")
    print(vocab)
    # getData("./mp3_release/data/train", vocab)
    getData(vocab, "./mp3_release/data/train", 500)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
