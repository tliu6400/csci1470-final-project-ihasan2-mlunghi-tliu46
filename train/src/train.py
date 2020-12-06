import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
from transformer import Transformer
from preprocess import get_data
# import compute_metrics
import matplotlib.pyplot as plt
import pickle

def train(model, train_inputs, train_labels, padding_index):
    #utilizes the Dataset API to efficiently process and batch train data
    train_inputs = tf.data.Dataset.from_tensor_slices(train_inputs)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    train_inputs = train_inputs.batch(model.batch_sz)
    train_inputs = train_inputs.prefetch(model.batch_sz)
    train_labels = train_labels.batch(model.batch_sz)
    train_labels = train_labels.prefetch(model.batch_sz)

    loss_over_time = []

    #iterates through data and executes training
    for inputs_batch, labels_batch in zip(train_inputs, train_labels):
        with tf.GradientTape() as tape:
            probs = model.call(inputs_batch, labels_batch[:, :-1])
            loss = model.loss(probs, labels_batch[:, 1:], labels_batch[:, 1:] != padding_index)
            print("Train Loss {}".format(loss.numpy()))
            loss_over_time.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return max(loss_over_time)




    # shuffle the indices of the input sentences
    # indices = tf.range(train_labels.shape[0])
    # indices = tf.random.shuffle(indices)
    # train_inputs = tf.gather(train_inputs, indices)
    # train_labels = tf.gather(train_labels, indices)

    # # run batches of the data
    # for i in range(0, train_labels.shape[0], model.batch_sz):
    #     # define an end in case we are on last batch
    #     end = min(i + model.batch_sz, train_labels.shape[0])
    #     with tf.GradientTape() as tape:
    #         # call model on a batch of inputs
    #         probs = model.call(train_inputs[i:end], train_labels[i:end, :-1])
    #         # calculate loss on a batch of inputs using
    #         loss = model.loss(probs, train_labels[i:end, 1:], train_labels[i:end, 1:] != padding_index)
    #         print(loss)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels, padding_index):
    #utilizes the Dataset API to efficiently process and batch test data
    test_inputs = tf.data.Dataset.from_tensor_slices(test_inputs)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    test_inputs = test_inputs.batch(model.batch_sz)
    test_inputs = test_inputs.prefetch(model.batch_sz)
    test_labels = test_labels.batch(model.batch_sz)
    test_labels = test_labels.prefetch(model.batch_sz)

    loss_over_time = []

    #computes the loss over the test data
    for inputs_batch, labels_batch in zip(test_inputs, test_labels):
        probs = model.call(inputs_batch, labels_batch[:, :-1])
        loss = model.loss(probs, labels_batch[:, 1:], labels_batch[:, 1:] != padding_index)

        loss_over_time.append(loss.numpy())

        print("Test Loss {}".format(loss.numpy()))

    return min(loss_over_time)

def main():
    #parses the arguments passed into the program via the command line
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--model", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    #error checks the model argument
    if args.model is None or args.model not in {"TAGGER", "GENERATOR"}:
        print("--model parameter must be \"TAGGER\" or \"GENERATOR\"")
        exit()

    #executes the program using the right model
    if args.model == "TAGGER":
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/entagged_parallel.train.en.P_0", "../../data/train.tagged.P_0", "../../data/entagged_parallel.test.en.P_0", "../../data/test.tagged.P_0")
    else:
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/train.tagged.P_9", "../../data/entagged_parallel.train.en.P_9", "../../data/entagged_parallel.test.en.P_9", "../../data/test.tagged.P_9")

    padding_index = vocab["*PAD*"]

    #initializes the model
    if args.load is not None:
        model = tf.saved_model.load(args.load)
    else:
        model = Transformer(vocab, reverse_vocab)


    train_loss, test_loss = [], [0]

    #trains the model
    for i in range(1, 21):
        print("----------Starting training epoch {}----------".format(i))
        train_loss.append(train(model, train_inputs, train_labels, padding_index))
        test_loss.append(test(model, test_inputs, test_labels, padding_index))


    #write the train and test loss arrays into a file_name
    with open("train_loss.txt", "wb") as fp:
        pickle.dump(train_loss, fp)
    with open("test_loss.txt", "wb") as fp:
        pickle.dump(test_loss, fp)


    # to retrieve this later we will use
    # with open("test.txt", "rb") as fp:   # Unpickling into object b
    #     b = pickle.load(fp)



    # with open("test_loss.txt", "rb") as fp:
    #     test_loss = pickle.load(fp)
    #
    # with open("train_loss.txt", "rb") as fp:
    #     train_loss = pickle.load(fp)

    x_tr = range(len(train_loss))
    x_te = range(len(test_loss))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    # ax1.scatter(x_tr, train_loss, s=10, c='b', marker='s', label='training loss')
    # ax1.scatter(x_te, test_loss, s=10, c='r', marker='o', label='testing loss')
    # plt.legend(loc='upper right');
    #
    # plt.show()


    #sample model
    idx = random.choice(range(len(test_inputs)-1))
    print("Input sentence: {}".format([reverse_vocab[test_inputs[idx, i]] for i in range(len(test_inputs[idx]))]))
    print("Label sentence: {}".format([reverse_vocab[test_labels[idx, i]] for i in range(len(test_labels[idx]))]))
    probs = model.call(tf.expand_dims(test_inputs[idx], axis=0), tf.expand_dims(test_labels[idx, :-1], axis=0))
    output_sentence = tf.math.argmax(probs[0, :, :], axis=1)
    print("Output sentence: {}".format([reverse_vocab[output_sentence[i].numpy()] for i in range((len(output_sentence)))]))


    # from random import randint
    #
    # start = randint(0,len(test_inputs)-250)
    #
    # collected_outputs, label_sentences = [], []
    # for i in range(start, start+50):
    #
    #     lab = [reverse_vocab[test_labels[i, j]] for j in range(len(test_labels[i]))]
    #     label_sentences.append(lab)
    #     probs = model.call(tf.expand_dims(test_inputs[i], axis=0), tf.expand_dims(test_labels[i, :-1], axis=0))
    #     output_sentence = tf.math.argmax(probs[0, :, :], axis=1)
    #     output_sentence = [reverse_vocab[output_sentence[j].numpy()] for j in range((len(output_sentence)))]
    #     collected_outputs.append(output_sentence)
    #
    #
    # #Computes Rouge metrics
    # fluid_generated_sentences = []
    # for array in collected_outputs:
    #     fluid_generated_sentences.append(' '.join(array))
    #
    # fluid_labels = []
    # for array in label_sentences:
    #     fluid_labels.append(' '.join(array))
    #
    # generated_corpus_output = '.\n'.join(fluid_generated_sentences)
    # label_corpus = '.\n'.join(fluid_labels)

    # compute_metrics.main(generated_corpus_output, collected_outputs, label_corpus, label_sentences)
    #

    #save the model
    if args.save is not None:
        tf.saved_model.save(model, args.save)

if __name__ == '__main__':
    main()
