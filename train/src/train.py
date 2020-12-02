import os
import sys
import argparse
import tensorflow as tf
from transformer import Transformer
from preprocess import get_data

def train(model, train_inputs, train_labels, padding_index, r_v):
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

    for inputs_batch, labels_batch in zip(train_inputs, train_labels):
        with tf.GradientTape() as tape:
            probs = model.call(inputs_batch, labels_batch[:, :-1])                
            loss = model.loss(probs, labels_batch[:, 1:], labels_batch[:, 1:] != padding_index)
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def main():
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--model", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    if args.model is None or args.model not in {"TAGGER", "GENERATOR"}:
        print("--model parameter must be \"TAGGER\" or \"GENERATOR\"")
        exit()

    # Preprocess
    if args.model == "TAGGER":
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/entagged_parallel.train.en.P_0", "../../data/train.tagged.P_0", "../../data/entagged_parallel.test.en.P_0", "../../data/test.tagged.P_0")
    else:
        train_inputs, train_labels, test_inputs, test_labels, vocab, reverse_vocab = get_data("../../data/train.tagged.P_9", "../../data/entagged_parallel.train.en.P_9", "../../data/entagged_parallel.test.en.P_9", "../../data/test.tagged.P_9")

    # train_inputs = tf.convert_to_tensor(train_inputs)
    # train_labels = tf.convert_to_tensor(train_labels)
    # test_inputs = tf.convert_to_tensor(test_inputs)
    # test_labels = tf.convert_to_tensor(test_labels)

    padding_index = vocab["*PAD*"]

    # Initialize model
    if args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        model = Transformer(len(vocab))

    # Train model
    for i in range(1, 10):
        print("Epoch {}".format(i))
        train(model, train_inputs, train_labels, padding_index, reverse_vocab)

    print("\nInput sentence is \n", [reverse_vocab[test_inputs[1, i].numpy()] for i in range(len(test_inputs[0]))])
    print("\nLabel sentence is \n", [reverse_vocab[test_labels[1, i].numpy()] for i in range(len(test_labels[0]))])
    probs = model.call(tf.expand_dims(test_inputs[1], axis=0), tf.expand_dims(test_labels[1, :-1], axis=0))
    output_sentence = tf.math.argmax(probs[0,:,:], axis=1)
    print("\nOutput sentence is \n", [reverse_vocab[output_sentence[i].numpy()] for i in range((output_sentence.shape[0]))])

    # Save model
    if args.save is not None:
        model.save(args.save)


if __name__ == '__main__':
    main()

