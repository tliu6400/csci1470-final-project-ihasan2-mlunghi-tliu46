# import argparse
# import tensorflow as tf
# from preprocess import convert_to_id

# def pad_sentence(sentence):
#     max_length = 32
#     padded_input = sentence[:max_length]
#     padded_input += ["*STOP*"] + ["*PAD*"] * (max_length - len(padded_input))
#     return padded_input

# def convert_tags(sentence):
#     for i in len(sentence):
#         if "P0" in sentence[i]:
#             sentence[i].replace("P0", "P9")

# def main():
#     parser = argparse.ArgumentParser("Train model")
#     parser.add_argument("--tagger", type=str)
#     parser.add_argument("--generator", type=str)
#     parser.add_argument("--input", type=str)
#     args = parser.parse_args()

#     if args.tagger is None or args.generator is None or args.input is None:
#         print("--tagger, --generator, and --input parameters are required")
#         exit()

#     # Load models
#     tagger = tf.saved_model.load(args.tagger)
#     generator = tf.saved_model.load(args.generator)

#     # Preprocess input
#     processed = args.input.split()
#     # TODO: tokenize properly 
#     processed = pad_sentence(processed)
#     processed = convert_to_id(tagger.vocab, processed)

#     # Pass input through pipeline
#     tagged = tagger.sample(processed)
#     convert_tags(tagged)
#     generated = generator.sample(converted)

#     # Print generated statment
#     print("----------POLITENESS TRANSFER MODEL: IMPOLITE --> POLITE----------")
#     print("Impolite Input: {}".format(args.input))
#     print("Polite Output: {}".format(generated))

# if __name__ == '__main__':
#     main()