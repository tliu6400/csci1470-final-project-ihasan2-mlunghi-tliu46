import os
import sys
import argparse
import tensorflow as tf
from transformer import Transformer
from preprocess import get_data

def train(model, train_inputs, train_labels, padding_index, r_v):
    # shuffle the indices of the input sentences
    indices = tf.range(train_labels.shape[0])
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    # run batches of the data
    for i in range(0, train_labels.shape[0], model.batch_sz):
        # define an end in case we are on last batch
        end = min(i + model.batch_sz, train_labels.shape[0])
        with tf.GradientTape() as tape:
            # call model on a batch of inputs
            probs = model.call(train_inputs[i:end], train_labels[i:end, :-1])                
            # calculate loss on a batch of inputs using 
            loss = model.loss(probs, train_labels[i:end, 1:], train_labels[i:end, 1:] != padding_index)
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
        train_inputs, train_labels, vocab, reverse_vocab = get_data("../../data/entagged_parallel.train.en.catcher", "../../data/train.tagged.catcher")
    else:
        train_inputs, train_labels, vocab, reverse_vocab = get_data("../../data/train.tagged.romeo-juliet", "../../data/entagged_parallel.train.en.romeo-juliet")

    train_inputs = tf.convert_to_tensor(train_inputs)
    train_labels = tf.convert_to_tensor(train_labels)

    padding_index = vocab["*PAD*"]

    # Initialize model
    if args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        model = Transformer(len(vocab))

    # Train model
    for i in range(1, 500):
        print("Epoch {}".format(i))
        train(model, train_inputs, train_labels, padding_index, reverse_vocab)
    print("Input sentence is ", [reverse_vocab[train_inputs[0, i].numpy()] for i in range(len(train_inputs[0]))])
    print("Label sentence is ", [reverse_vocab[train_labels[0, i].numpy()] for i in range(len(train_labels[0]))])
    probs = model.call(tf.expand_dims(train_inputs[0], axis=0), tf.expand_dims(train_labels[0, :-1], axis=0))
    output_sentence = tf.math.argmax(probs[0,:,:], axis=1)
    print("Output sentence is ", [reverse_vocab[output_sentence[i].numpy()] for i in range((output_sentence.shape[0]))])

    # Save model
    if args.save is not None:
        model.save(args.save)


if __name__ == '__main__':
    main()

