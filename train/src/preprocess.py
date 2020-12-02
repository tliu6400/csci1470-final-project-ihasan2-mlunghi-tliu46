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

def build_vocab(train_input_sentences, train_labels_sentences, test_input_sentences, test_labels_sentences):
    tokens = []
    for s in train_input_sentences: tokens.extend(s)
    for s in train_labels_sentences: tokens.extend(s)
    for s in test_input_sentences: tokens.extend(s)
    for s in test_labels_sentences: tokens.extend(s)
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

def get_data(train_inputs_file, train_labels_file, test_inputs_file, test_labels_file):
    # Read files
    train_inputs = read_data(train_inputs_file)
    train_labels = read_data(train_labels_file)
    test_inputs = read_data(test_inputs_file)
    test_labels = read_data(test_labels_file)

    # Pad sentences
    train_inputs, train_labels = pad_corpus(train_inputs, train_labels)
    test_inputs, test_labels = pad_corpus(test_inputs, test_labels)

    # Build vocab
    vocab = build_vocab(train_inputs, train_labels, test_inputs, test_labels)

    # Convert to ids
    train_inputs = convert_to_id(vocab, train_inputs)
    train_labels = convert_to_id(vocab, train_labels)
    test_inputs = convert_to_id(vocab, test_inputs)
    test_labels = convert_to_id(vocab, test_labels)

    reverse_vocab = {v: k for k, v in vocab.items()}

    train_inputs = tf.data.Dataset.from_tensors(train_inputs)
    train_labels = tf.data.Dataset.from_tensors(train_labels)
    test_inputs = tf.data.Dataset.from_tensors(test_inputs)
    test_labels = tf.data.Dataset.from_tensors(test_labels)

    train_inputs = train_inputs.batch(640)
    train_inputs = train_inputs.prefetch(1)

    train_labels = train_labels.batch(640)
    train_labels = train_labels.prefetch(1)

    test_inputs = test_inputs.batch(640)
    test_inputs = test_inputs.prefetch(1)

    test_labels = test_labels.batch(640)
    test_labels = test_labels.prefetch(1)

    return train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab
