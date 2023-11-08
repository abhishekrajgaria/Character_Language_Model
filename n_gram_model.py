import time
import math
from custom_data_loader import *
from mp3_release.scripts import utils


def convert_line2char(line, vocab):
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append("<unk>")
        else:
            line_data.append(charac)
    return line_data


def convert_files2char(files, vocab):
    data = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            toks = convert_line2char(line, vocab)
            data.append(toks)
    return data


def convert_chars_to_str(chars_list):
    return "".join(chars_list)


def genTriAndFourGramDict(vocab_filename, data_folder):
    vocab, _ = getVocab(vocab_filename)
    data_files = utils.get_files(data_folder)
    data = convert_files2char(data_files, vocab)

    dict3 = {}
    dict4 = {}
    pad_char = "[PAD]"
    for line in data:
        trigram = [pad_char, pad_char, pad_char]
        fourgram = [pad_char, pad_char, pad_char]
        for char in line:
            fourgram.append(char)
            trigram_str = convert_chars_to_str(trigram)
            fourgram_str = convert_chars_to_str(fourgram)
            if trigram_str not in dict3:
                dict3[trigram_str] = 1
            else:
                dict3[trigram_str] += 1
            if fourgram_str not in dict4:
                dict4[fourgram_str] = 1
            else:
                dict4[fourgram_str] += 1

            trigram.pop(0)
            trigram.append(char)
            fourgram.pop(0)

        trigram_str = convert_chars_to_str(trigram)
        if trigram_str not in dict3:
            dict3[trigram_str] = 1
        else:
            dict3[trigram_str] += 1

    print(len(dict3), len(dict4))
    return dict3, dict4


def getPerplexity(vocab_filename, train_data_folder, test_data_folder):
    vocab, _ = getVocab(vocab_filename)
    vocab_len = len(vocab.keys())
    test_data_files = utils.get_files(test_data_folder)
    test_data = convert_files2char(test_data_files, vocab)

    print("test_data_generated")

    trigram_cnt, fourgram_cnt = genTriAndFourGramDict(vocab_filename, train_data_folder)

    print("trigram and fourgram data_generated")

    data_cnt = len(test_data)
    perplexity = 0
    pad_char = "[PAD]"
    for line in test_data:
        trigram = [pad_char, pad_char, pad_char]
        fourgram = [pad_char, pad_char, pad_char]
        n = len(line)
        log_prob_sum = 0
        for char in line:
            fourgram.append(char)
            trigram_str = convert_chars_to_str(trigram)
            fourgram_str = convert_chars_to_str(fourgram)
            numerator = 1
            denominator = vocab_len

            if trigram_str in trigram_cnt:
                denominator += trigram_cnt[trigram_str]

            if fourgram_str in fourgram_cnt:
                numerator += fourgram_cnt[fourgram_str]

            log_prob_sum += math.log2(numerator / denominator)

            trigram.pop(0)
            trigram.append(char)
            fourgram.pop(0)
        line_perplexity = (-1 * log_prob_sum) / n
        perplexity += 2**line_perplexity
    perplexity = perplexity / data_cnt

    print(perplexity)
    return perplexity


if __name__ == "__main__":
    start_time = time.time()
    # genTriAndFourGramDict("./mp3_release/data/vocab.pkl", "./mp3_release/data/train")
    getPerplexity(
        "./mp3_release/data/vocab.pkl",
        "./mp3_release/data/train",
        "./mp3_release/data/test",
    )
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
