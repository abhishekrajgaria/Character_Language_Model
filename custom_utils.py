def convert_sent2idx(sentences, vocab):
    data = []
    for sent in sentences:
        sent_data = []
        for charac in sent:
            if charac not in vocab.keys():
                sent_data.append(vocab["<unk>"])
            else:
                sent_data.append(vocab[charac])
        data.append(sent_data)
    return data
