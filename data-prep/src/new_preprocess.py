import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# this function handles the dirty work of the preprocessing
# Specifying it with a tsv and an output directory, it will populate the output directory with the appropriate information
def generate(source, dest, style1, style2):
    # convert data into a pandas dataframe
    data = pd.read_csv(source, sep="\t")
    # collect all sentences where the style is one of our two desired styles
    style1_subset, style2_subset = data[data["style"] == style1], data[data["style"] == style2]

    id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first = stat_list(style1_subset,style1)
    id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second = stat_list(style2_subset,style2)

    
    tags1 = relative_tag_generator(style1, id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first,
                            style2, id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second)

    tags2 = relative_tag_generator(style2, id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second,
                                    style1, id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first)


    # write the json files with the relevant tags to the output directory
    with open(f"{dest}/{style1}_tags.json", "w") as f:
        json.dump(tags1, f)
    with open(f"{dest}/{style2}_tags.json", "w") as f:
        json.dump(tags2, f)

    print("GENERATING TAGGED DATA")

    create_datasets(style1_subset, dest, tags1, style1)
    create_datasets(style2_subset, dest, tags2, style2)

def create_datasets(s, dest, tags, style):


    # Given a sentence and a dictionary of taggable words, replace all taggable words
    def tag_sentence(sent, tag_dict, tag_token):
        i = 0
        sentence = sent.split()
        tagged_sent = []
        while i < len(sentence):
            # look for words or bigrams
            bigram = " ".join(sentence[i: i + 2])
            if bigram in tag_dict:
                tagged_sent.append(f"[{tag_token}{i}]")
                i += 2
            elif sentence[i] in tag_dict:
                tagged_sent.append(f"[{tag_token}{i}]")
                i += 1
            else:
                tagged_sent.append(sentence[i])
                i += 1
        return " ".join(tagged_sent)


    def tag_and_dump(split):

        original_sentences, tagged_sentences = [], []
        if split == "train":
            data_in = s[s["split"]!= "test"]
        else:
            data_in = s[s["split"] == split]

        for _,r in data_in.iterrows():
            original = r['txt'].strip().replace("\n", "")
            original_sentences.append(original)
            tagged_sentences.append(tag_sentence(original, tags, style).strip().replace("\n", ""))

        with open(f"{dest}/entagged_parallel.{split}.en.{style}", "w") as original_out,\
             open(f"{dest}/{split}.tagged.{style}", "w") as tagged_out:

            for original, tagged in zip(original_sentences, tagged_sentences):
                if style in tagged:
                    ### ONLY WRITE OUT THE Tagged DATA
                    original_out.write(f"{original.strip()}\n")
                    tagged_out.write(f"{tagged.strip()}\n")


    tag_and_dump(split='train')
    tag_and_dump(split='test')

    return



def stat_list(sentences, style, ngram_range =(1,2)):

    sentences = sentences['txt']

    # generate tfidf stats for this style
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names()

    id_to_word = {i: feature_names[i] for i in range(len(vectorizer.get_feature_names()))}
    word_to_id = {id_to_word[k]: k for k in id_to_word}
    X = np.asarray(X.mean(axis=0)).squeeze()
    idf = vectorizer.idf_
    word_to_idf = dict(zip(feature_names, idf))

    # counting the ocurrence of each of these words or phrases
    count_vectorizer = CountVectorizer(ngram_range=ngram_range)
    counted = count_vectorizer.fit_transform(sentences)
    fn = count_vectorizer.get_feature_names()
    Y = np.asarray(counted.sum(axis=0))
    word_2_id = {fn[i]: i for i in range(len(count_vectorizer.get_feature_names()))}
    word_count = {}
    for w in word_2_id:
        word_count[w] = Y[0, word_2_id[w]]


    return id_to_word, word_to_id, X, word_to_idf, word_count


# determine the tags that are most distinctive to each style
def relative_tag_generator(style_m, id_to_word_m, word_to_id_m, X_m, word_to_idf_m, word_count_m,
                            style_r, id_to_word_r, word_to_id_r, X_r, word_to_idf_r, word_count_r):
    output = []

    for word in word_to_id_m.keys():
        # consider words that appear more than once and their relative stats for each style
        if word_count_m[word] >=2 and word in word_to_id_r:
                res = {}
                res["word"] = word
                res["freq"] = word_count_m[word]
                res[f"{style_m}_mean_tfidf"] = X_m[word_to_id_m[word]]
                res[f"{style_r}_mean_tfidf"] = X_r[word_to_id_r[word]]
                res[f"{style_m}_idf"] = word_to_idf_m[word]
                res[f"{style_r}_idf"] = word_to_idf_r[word]
                output.append(res)
    output = pd.DataFrame(output)

    # generate dictionary of phrases that are important in main compared to relative
    sm_sr = f"{style_m}_over_{style_r}"
    sr_sm = f"{style_r}_over_{style_m}"
    output[sm_sr] = output[f"{style_m}_mean_tfidf"] / output[f"{style_r}_mean_tfidf"]
    output[sr_sm] = 1 / output[sm_sr]

    # support for small values
    output[f"{style_m}_tag"] = (output[sm_sr] / output[sm_sr].sum()) ** 0.75

    # turn into probability 
    output[f"{style_m}_tag"] = output[f"{style_m}_tag"] / output[f"{style_m}_tag"].sum()

    # assign a percentile ranking
    output.sort_values(by=f"{style_m}_tag", ascending=False, inplace=True)
    output['rank'] = output[f"{style_m}_tag"].rank(pct=True)
  
    # the important words/phrases should be the top 90%
    important_phrases = output[output["rank"] >= 0.9]

    # make distribution
    important_phrases["score"] = (important_phrases["rank"] - 0.9) / (1 - 0.9)
    
    tags= {}
    for i, r in important_phrases.iterrows():
        tags[r["word"]] = r["score"]
    return tags



if __name__ == '__main__':


    print("Starting Preprocessing!")
    # path to the tsv file
    tsv_path = "../../data/politeness.tsv"
    # path to the destination of the output files
    output_path = "../../data/"

    #generate tags
    generate(source=tsv_path, dest=output_path, style1="P_0", style2="P_9")

    print("preprocessing over")

