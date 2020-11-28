import os
import sys
import argparse
import tensorflow as tf
from transformer import Transformer
from preprocess import get_data

def train(model, train_inputs, train_labels, padding_index):
    for i in range(0, train_labels.shape[0], model.batch_sz):
        end = min(i + model.batch_sz, train_labels.shape[0])
        with tf.GradientTape() as tape:
            probs = model.call(train_inputs[i:end], train_labels[i:end, :-1])
            loss = model.loss(probs, train_labels[i:end, 1:], train_labels[i:end, 1:] != padding_index)
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
        train_inputs, train_labels, vocab = get_data("../../data/entagged_parallel.train.en.catcher", "../../data/train.tagged.catcher")
    else:
        train_inputs, train_labels, test_inputs, test_labels, vocab = get_data("../../data/train.tagged.romeo-juliet", "../../data/entagged_parallel.train.en.romeo-juliet")

    padding_index = vocab["*PAD*"]

    # Initialize model
    if args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        model = Transformer(len(vocab))

    # Train model
    for i in range(1, 6):
        print("Epoch {}".format(i))
        train(model, train_inputs, train_labels, padding_index)

    # Save model
    if args.save is not None:
        model.save(args.save)


if __name__ == '__main__':
    main()

