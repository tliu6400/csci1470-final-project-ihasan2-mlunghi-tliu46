import numpy as np
import tensorflow as tf

pad_token = "*PAD*"
start_token = "*START*"
stop_token = "*STOP*"
unk_token = "*UNK*"

def read_data(file_name):
    text = []
    with open(file_name, 'r') as data_file:
        for line in data_file: text.append(line.split())
    return text

def build_vocab(input_sentences, labels_sentences):
    tokens = []
    for s in input_sentences: tokens.extend(s)
    for s in labels_sentences: tokens.extend(s)
    all_words = sorted(list(set([pad_token, start_token, stop_token, unk_token] + tokens)))
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab

def pad_corpus(input_sentences, labels_sentences):
    max_length = max([len(s) for s in input_sentences] + [len(s) for s in labels_sentences]) + 1
    input_padded_sentences = []
    for s in input_sentences:
        padded_input = s[:max_length]
        padded_input += [stop_token] + [pad_token] * (max_length - len(padded_input) - 1)
        input_padded_sentences.append(padded_input)
    labels_padded_sentences = []
    for s in labels_sentences:
        padded_label = s[:max_length]
        padded_label = [start_token] + padded_label + [stop_token] + [pad_token] * (max_length - len(padded_label) - 1)
        labels_padded_sentences.append(padded_label)
    return input_padded_sentences, labels_padded_sentences

def convert_to_id(vocab, sentences):
    return np.stack([[vocab[word] if word in vocab else vocab[unk_token] for word in sentence] for sentence in sentences])

def get_data(inputs_file, labels_file):
    # Read files
    inputs = read_data(inputs_file)
    labels = read_data(labels_file)

    # Pad sentences
    inputs, labels = pad_corpus(inputs, labels)

    # Build vocab
    vocab = build_vocab(inputs, labels)

    # Convert to ids
    inputs = convert_to_id(vocab, inputs)
    labels = convert_to_id(vocab, labels)

    # # Shuffle sentences
    # indices = tf.range(inputs.shape[0])
    # indices = tf.random.shuffle(indices)
    # inputs = tf.gather(inputs, indices)
    # labels = tf.gather(labels, indices)

    # # Split into test and train
    # train_inputs = inputs[:int(inputs.shape[0]*0.8)]
    # test_inputs = inputs[int(inputs.shape[0]*0.8):]
    # train_labels = labels[:int(labels.shape[0]*0.8)]
    # test_labels = labels[int(labels.shape[0]*0.8):]

    return inputs, labels, vocab