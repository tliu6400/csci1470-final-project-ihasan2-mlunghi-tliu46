import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
from transformer import Transformer
from preprocess import get_data
import compute_metrics
import matplotlib.pyplot as plt
import pickle

def train(model, train_inputs, train_labels, padding_index):
    # Utilize Dataset API to efficiently process and batch train data
    train_inputs = tf.data.Dataset.from_tensor_slices(train_inputs)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    train_inputs = train_inputs.batch(model.batch_sz)
    train_inputs = train_inputs.prefetch(model.batch_sz)
    train_labels = train_labels.batch(model.batch_sz)
    train_labels = train_labels.prefetch(model.batch_sz)

    loss_over_time = []

    # Iterate through data and execute training
    for inputs_batch, labels_batch in zip(train_inputs, train_labels):
        with tf.GradientTape() as tape:
            probs = model.call(inputs_batch, labels_batch[:, :-1])
            loss = model.loss(probs, labels_batch[:, 1:], labels_batch[:, 1:] != padding_index)
            loss_over_time.append(loss.numpy())
            print("Train Loss {}".format(loss.numpy()))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return sum(loss_over_time) / len(loss_over_time)

def test(model, test_inputs, test_labels, padding_index):
    # Utilize Dataset API to efficiently process and batch train data
    test_inputs = tf.data.Dataset.from_tensor_slices(test_inputs)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    test_inputs = test_inputs.batch(model.batch_sz)
    test_inputs = test_inputs.prefetch(model.batch_sz)
    test_labels = test_labels.batch(model.batch_sz)
    test_labels = test_labels.prefetch(model.batch_sz)

    loss_over_time = []

    # Compute loss over the test data
    for inputs_batch, labels_batch in zip(test_inputs, test_labels):
        probs = model.call(inputs_batch, labels_batch[:, :-1])
        loss = model.loss(probs, labels_batch[:, 1:], labels_batch[:, 1:] != padding_index)
        loss_over_time.append(loss.numpy())
        print("Test Loss {}".format(loss.numpy()))
    return sum(loss_over_time) / len(loss_over_time)

def main():
    # Parses command line arguments
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--model", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--visualize", type=bool)
    args = parser.parse_args()

    # Error checks --model argument
    if args.model is None or args.model not in {"TAGGER", "GENERATOR"}:
        print("--model parameter must be \"TAGGER\" or \"GENERATOR\"")
        exit()

    # Loads appropriate data
    if args.model == "TAGGER":
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/entagged_parallel.train.en.P_0", "../../data/train.tagged.P_0", "../../data/entagged_parallel.test.en.P_0", "../../data/test.tagged.P_0")
    else:
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/train.tagged.P_9", "../../data/entagged_parallel.train.en.P_9", "../../data/test.tagged.P_9", "../../data/entagged_parallel.test.en.P_9")

    padding_index = vocab["*PAD*"]

    # Initialize the model
    if args.load is not None:
        model = tf.saved_model.load(args.load)
    else:
        model = Transformer(vocab, reverse_vocab)

    # Train the model
    train_loss, test_loss = [], []
    for i in range(1, 1):
        print("----------Starting training epoch {}----------".format(i))
        train_loss.append(train(model, train_inputs, train_labels, padding_index))
        test_loss.append(test(model, test_inputs, test_labels, padding_index))

    # Visualize train and test loss of model
    if args.visualize:
        previous_train_loss = []
        previous_test_loss = []
        if args.load is not None:
            with open("../../evaluation/loss_files/" + args.model + "_train_loss.pickle", "rb") as fp:
                previous_train_loss = pickle.load(fp)
            with open("../../evaluation/loss_files/" + args.model + "_test_loss.pickle", "rb") as fp:
                previous_test_loss = pickle.load(fp)
        train_loss = previous_train_loss + train_loss
        test_loss = previous_test_loss + test_loss
        with open("../../evaluation/loss_files/" + args.model + "_train_loss.pickle", "wb") as fp:
            pickle.dump(train_loss, fp)
        with open("../../evaluation/loss_files/" + args.model + "_test_loss.pickle", "wb") as fp:
            pickle.dump(test_loss, fp)

        x_tr = range(len(train_loss))
        x_te = range(len(test_loss))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.title("Maximal loss across Epochs of Training GENERATOR Model")
        plt.xlabel("Epoch #")
        plt.ylabel("Mean Loss per Epoch")
        ax1.scatter(x_tr, train_loss, s=10, c='b', marker='s', label='training loss')
        ax1.scatter(x_te, test_loss, s=10, c='r', marker='o', label='testing loss')
        plt.legend(loc='upper right');
        plt.show()

    # Sample model
    idx = random.choice(range(len(test_inputs)-1))
    print("Input sentence: {}".format([reverse_vocab[test_inputs[idx, i]] for i in range(len(test_inputs[idx]))]))
    print("Label sentence: {}".format([reverse_vocab[test_labels[idx, i]] for i in range(len(test_labels[idx]))]))
    probs = model.call(tf.expand_dims(test_inputs[idx], axis=0), tf.expand_dims(test_labels[idx, :-1], axis=0))
    output_sentence = tf.math.argmax(probs[0, :, :], axis=1)
    print("Output sentence: {}".format([reverse_vocab[output_sentence[i].numpy()] for i in range((len(output_sentence)))]))

    # Select 50 random sentences to evaluate metrics on
    start = random.randint(0,len(test_inputs)-250)
    collected_outputs, label_sentences = [], []
    for i in range(start, start+50):
        lab = [reverse_vocab[test_labels[i, j]] for j in range(len(test_labels[i]))]
        label_sentences.append(lab)
        probs = model.call(tf.expand_dims(test_inputs[i], axis=0), tf.expand_dims(test_labels[i, :-1], axis=0))
        output_sentence = tf.math.argmax(probs[0, :, :], axis=1)
        output_sentence = [reverse_vocab[output_sentence[j].numpy()] for j in range((len(output_sentence)))]
        collected_outputs.append(output_sentence)

    # Compute ROUGE and BLEU metrics
    fluid_generated_sentences = []
    for array in collected_outputs:
        fluid_generated_sentences.append(' '.join(array))
    fluid_labels = []
    for array in label_sentences:
        fluid_labels.append(' '.join(array))
    generated_corpus_output = '.\n'.join(fluid_generated_sentences)
    label_corpus = '.\n'.join(fluid_labels)
    compute_metrics.main(generated_corpus_output, collected_outputs, label_corpus, label_sentences)

    # Save the model
    if args.save is not None:
        tf.saved_model.save(model, args.save)

if __name__ == '__main__':
    main()
