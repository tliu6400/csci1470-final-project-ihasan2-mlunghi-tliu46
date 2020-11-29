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

            # Trying to see what max probs the model gave us
            if i == 0:
                # check an output sentence


                converted_sentence = tf.math.argmax(probs[0,:,:], 1)
                # print("converted sentence is: ", converted_sentence)

                # print("original sentence is, ", train_inputs[0,:])

                original_sentence = train_inputs[0]
                print("\n\n\n original sentence is ", [r_v[original_sentence[i].numpy()] for i in range(len(original_sentence))])
                # print("original sentence in word form ", [r_v[train_inputs[0,:][i]] for i in range(len(train_inputs[0,:])])

                # print("was meant to tag as , ", train_labels[0,:])
                lab = train_labels[0]
                print("\n\n\ntagged sentence is ", [r_v[lab[i].numpy()] for i in range(len(lab))])

                print("\n\n\n but what we actually got is ", [r_v[converted_sentence[i].numpy()] for i in range(len(converted_sentence)) ])


                # print("AND THE WORD IS ", r_v[tf.math.argmax(probs[0,:,:],1)[0].numpy()])
                # print("just to be clear the word i printed is ", tf.math.argmax(probs[0,:,:], 1)[0])

                # print("what the hell is 2 ",  r_v[2])
                




            # calculate loss on a batch of inputs using 
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
        train_inputs, train_labels, vocab, reverse_vocab = get_data("../../data/entagged_parallel.train.en.catcher", "../../data/train.tagged.catcher")
    else:
        train_inputs, train_labels, vocab, reverse_vocab = get_data("../../data/train.tagged.romeo-juliet", "../../data/entagged_parallel.train.en.romeo-juliet")


    # print(train_inputs[0:10])
    # print(train_labels[0:10])

    train_inputs = tf.convert_to_tensor(train_inputs)
    train_labels = tf.convert_to_tensor(train_labels)

    padding_index = vocab["*PAD*"]

    # Initialize model
    if args.load is not None:
        model = tf.keras.models.load_model(args.load)
    else:
        model = Transformer(len(vocab))

    # Train model
    for i in range(1, 2):
        print("Epoch {}".format(i))
        train(model, train_inputs, train_labels, padding_index, reverse_vocab)

    # Save model
    if args.save is not None:
        model.save(args.save)


if __name__ == '__main__':
    main()

