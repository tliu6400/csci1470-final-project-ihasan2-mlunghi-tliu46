"""Generates tags
Usage:
    compute_metrics.py [options]

Options:
    --generated_data=<str>                        Path to generated data
    --labels=<str>                         Path to the labels
"""
# from docopt import docopt
from PyRouge.pyrouge import Rouge
# import bleu

def read_data(file_name):
    text = []
    with open(file_name, 'r') as data_file:
        for line in data_file: text.append(line.split())
    #text is now an array of arrays of strings
    sentences = []
    for array in text:
        sentences.append(' '.join(array))
    #new_text is now an array of strings
    #formatted_text is one giant string
    corpus = '.\n'.join(new_text)
    #returns corpus and arrays of strings
    return corpus, sentences

def compute_rouge(generated_data, labels):
    rouge = Rouge()
    [precision, recall, f_score] = rouge.rouge_l(generated_data, labels)
    return (precision, recall, f_score)

def compute_bleu(generated_data, labels):

    # Note: this is computed at the sentence level
    score = bleu.bleu_sentence_level(generated_data, labels)
    return score

def main(generated_corpus, generated_sentences, labels_corpus, labels_sentences):
    print("COMPUTING METRICS...")
    print("\n\n\n")

    #print rouge metrics
    print("ROUGE METRICS\n")
    precision, recall, f_score = compute_rouge(generated_corpus, labels_corpus)
    print("Precision: " + precision)
    print("Recall: " + recall)
    print("F_score: " + f_score)


    precision, recall, f_score = compute_rouge(generated_sentences, labels_sentences)
    print("Precision: " + precision)
    print("Recall: " + recall)
    print("F_score: " + f_score)






    print("\n")
    print("Score Explanation:")
    print("ROUGE-n recall=40% means that 40% of the n-grams in the reference summary" +
    "are also present in the generated summary.\nROUGE-n precision=40% means that 40%" +
    "of the n-grams in the generated summary are also present in the reference summary." +
    "\nROUGE-n F1-score=40% is more difficult to interpret, like any F1-score.")
    print("\n\n\n")

    #print blue metrics
    # print("BLUE METRICS\n")
    # blue_score = compute_bleu(generated_sentences, labels_sentences)
    # print("Score: " + blue_score)
    # print("\n\n\n")

# if __name__ == '__main__':
#
#     """
#     General Workflow:
#     1. Labels and generated data is read.
#     2. Each of the texts is an array of arrays of strings, which are joined into an array of strings and then into one string.
#     3. BLUE and ROUGE scores are computed and printed.
#
#     Resources:
#     ROUGE - https://github.com/pcyin/PyRouge
#     BLUE - https://github.com/neural-dialogue-metrics/BLEU
#     """
#
#     #pulls passed in file paths
#     # args = docopt(__doc__, version="")
#     generated_corpus, generated_sentences = read_data(args["--generated_data"])
#     labels_corpus, labels_sentences = read_data(args["--labels"])
#     main(generated_corpus, generated_sentences, labels_corpus, labels_sentences)
