from PyRouge.pyrouge import Rouge
from bleu import list_bleu

"""
General Workflow:
1. These functions ar ecalled from trian.py.
2. Each of the texts is an array of arrays of strings (sentences) or a just one large string (corpus).
3. BLUE and ROUGE scores are computed and printed.

Resources:
ROUGE - https://github.com/pcyin/PyRouge
BLUE - https://github.com/neural-dialogue-metrics/BLEU
"""

def read_data(file_name):
    text = []
    with open(file_name, 'r') as data_file:
        for line in data_file: text.append(line.split())
    # text is now an array of arrays of strings
    sentences = []
    for array in text:
        sentences.append(' '.join(array))
    # new_text is now an array of strings
    # formatted_text is one giant string
    corpus = '.\n'.join(new_text)
    return corpus, sentences

def compute_rouge(generated_data, labels):
    generated_data = [generated_data]
    labels = [labels]
    rouge = Rouge()
    [precision, recall, f_score] = rouge.rouge_l(generated_data,labels)
    return (precision, recall, f_score)

def compute_bleu(ref, hyp):
    # Note: this is computed at the sentence level
    new_ref = []
    new_hyp = []
    for r in ref:
        new_ref.append(' '.join(r))
    for r in hyp:
        new_hyp.append(' '.join(r))
    score = list_bleu([new_ref], new_hyp)
    return score

def main(generated_corpus, generated_sentences, labels_corpus, labels_sentences):
    print("COMPUTING METRICS...")
    print("\n\n\n")
    # Print rouge metrics
    print("ROUGE METRICS\n")
    try:
        precision, recall, f_score = compute_rouge(generated_corpus, labels_corpus)
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F_score: " + str(f_score))

        print("\n")
        print("Score Explanation:")
        print("ROUGE-n recall=40% means that 40% of the n-grams in the reference summary" +
        "are also present in the generated summary.\nROUGE-n precision=40% means that 40%" +
        "of the n-grams in the generated summary are also present in the reference summary." +
        "\nROUGE-n F1-score=40% is more difficult to interpret, like any F1-score.")
    except:
        print("ERROR: UNABLE TO ROUGE BLUE SCORE")
    print("\n\n\n")
    # Print blue metrics
    print("BLUE METRICS\n")
    try:
        score = compute_bleu(labels_sentences, generated_sentences)
        print("Blue Score: ")
        print(score)
    except:
        print("ERROR: UNABLE TO COMPUTE BLUE SCORE")
    print("\n\n\n")
