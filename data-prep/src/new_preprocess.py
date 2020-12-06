import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# this function handles the dirty work of the preprocessing
# Specifying it with a tsv and an output directory, it will populate the output directory with the appropriate information
def generate(source, dest, style1, style2):
    # convert data into a pandas dataframe
    data = pd.read_csv(source, sep="\t")
    # collect all sentences where the style is one of our two desired styles
    style1_subset, style2_subset = data[data["style"] == style1], data[data["style"] == style2]

    print("COMPUTING TF/IDF STATS")

    id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first = stat_list(style1_subset,style1)
    id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second = stat_list(style2_subset,style2)

    tags1 = relative_tag_generator(style1, id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first,
                            style2, id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second)

    tags2 = relative_tag_generator(style2, id_to_word_second, word_to_id_second, X_second, word_to_idf_second, word_count_second,
                                    style1, id_to_word_first, word_to_id_first, X_first, word_to_idf_first, word_count_first)




    print("LOADING JSON TAGS")

    # write the json files with the relevant tags to the output directory
    with open(f"{dest}/{style1}_tags.json", "w") as f:
        json.dump(tags1, f)
    with open(f"{dest}/{style2}_tags.json", "w") as f:
        json.dump(tags2, f)

    print("GENERATING TAGGED DATA")

    create_datasets(style1_subset, dest, tags1, style1)
    create_datasets(style2_subset, dest, tags2, style2)

def create_datasets(s, dest, tags, style):

    def tag_sentence(sent, tag_dict, tag_token,
                      pos_weight: int = 3,
                      max_pos_indicator: int = 20,
                      concat = True):
        """Given a sentence and a dictionary from
        tag_value to tag_probability, replaces all the words mw that are in the tag_dict
        with a probability tag_dict[mw]

        Arguments:
            sent {[str]}       -- [the given sentence]
            tag_dict {[dict]} -- [the tag dictionary]
            tag_token {[str]} -- [the taging token]
            dont_concat        -- [do not concat]

        Returns:
            [str] -- [the taged sentence]
        """
        i = 0
        sent = sent.split()
        tagged_sent = []
        prev_tag = False
        while i < len(sent):
            loc = min(i // pos_weight, max_pos_indicator)
            key_bi_gram = " ".join(sent[i: i + 2])
            key_tri_gram = " ".join(sent[i: i + 3])
            key_quad_gram = " ".join(sent[i: i + 4])

            if key_quad_gram in tag_dict and np.random.rand() < tag_dict[key_quad_gram]:
                if not concat or not prev_tag:
                    tagged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 4

            elif key_tri_gram in tag_dict and np.random.rand() < tag_dict[key_tri_gram]:
                if not concat or not prev_tag:
                    tagged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 3
            elif key_bi_gram in tag_dict and np.random.rand() < tag_dict[key_bi_gram]:
                if not concat or not prev_tag:
                    tagged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 2
            elif sent[i] in tag_dict and np.random.rand()< tag_dict[sent[i]]:
                if not concat or not prev_tag:
                    tagged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 1
            else:
                tagged_sent.append(sent[i])
                prev_tag = False
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
        #use this if file name isn't too long - ASK MATTEO
        # with open(f"{self.outpath}/en{self.target_lang}_parallel.{split}.en.{self.tag_token}", "w") as original_out,\
        #      open(f"{self.outpath}/en{self.target_lang}_parallel.{split}.{self.target_lang}.{self.tag_token}", "w") as tagged_out:
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

    # print("inside stat list the creation of word_to_id.keys() \n\n\n\n\n\n\n\n", sentences)

    sentences = sentences[sentences["split"]!= "test"]["txt"]

    # generate tfidf stats for first style
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names()




    id_to_word = {i: feature_names[i] for i in range(len(vectorizer.get_feature_names()))}
    word_to_id = {v: k for k, v in id_to_word.items()}
    X = np.asarray(X.mean(axis=0)).squeeze(0) # / num_docs
    idf = vectorizer.idf_
    word_to_idf = dict(zip(feature_names, idf))

    #count up instances
    count_vectorizer = CountVectorizer(ngram_range=ngram_range)
    counted = count_vectorizer.fit_transform(sentences)
    fn = count_vectorizer.get_feature_names()
    Y = np.asarray(counted.sum(axis=0)) # sum counts across sentences
    word_2_id = {feature_names[i]: i for i in range(len(count_vectorizer.get_feature_names()))}
    word_count = {}
    for w in word_2_id:
        word_count[w] = Y[0, word_2_id[w]]


    return id_to_word, word_to_id, X, word_to_idf, word_count

def relative_tag_generator(style_m, id_to_word_m, word_to_id_m, X_m, word_to_idf_m, word_count_m,
                            style_r, id_to_word_r, word_to_id_r, X_r, word_to_idf_r, word_count_r):
    report = []

    for word in word_to_id_m.keys():
        if word_count_m[word] >= 2 and word in word_to_id_r:
                # print('\n\n\n\ntest3')
                res = {}
                res["word"] = word
                res["freq"] = word_count_m[word]
                res[f"{style_m}_mean_tfidf"] = X_m[word_to_id_m[word]]



                res[f"{style_r}_mean_tfidf"] = X_r[word_to_id_r[word]]
                res[f"{style_m}_idf"] = word_to_idf_m[word]
                res[f"{style_r}_idf"] = word_to_idf_r[word]
                report.append(res)
    report = pd.DataFrame(report)

    # generate dictionary of phrases that are important in main compared to relative

    c1_over_c2 = f"{style_m}_over_{style_r}"
    c2_over_c1 = f"{style_r}_over_{style_m}"
    # tfidf_report["np_over_p"] = (tfidf_report["np_mean_tfidf"] / len(data_p_0)) / (tfidf_report["p_mean_tfidf"] /  len(data_p_9))
    report[c1_over_c2] = report[f"{style_m}_mean_tfidf"] / report[f"{style_r}_mean_tfidf"] #ratio of tf-idf in the two corpora

    report[c2_over_c1] = 1 / report[c1_over_c2]



    report[f"{style_m}_tag"] = (report[c1_over_c2] / report[c1_over_c2].sum()) ** 0.75
    # ^ add support for the small values

    report[f"{style_m}_tag"] = report[f"{style_m}_tag"] / report[f"{style_m}_tag"].sum()
    # ^ make a probability

    report.sort_values(by=f"{style_m}_tag", ascending=False, inplace=True)
    report['rank'] = report[f"{style_m}_tag"].rank(pct=True)
    # ^ assign percentile


    important_phrases = report[report["rank"] >= 0.9]
    # ^ only take phrases that clear the threshold (default: 0.9)

    important_phrases["score"] = (important_phrases["rank"] - 0.9) / (1 - 0.9)
    # ^ make a distribution again

    tags= {}
    for i, r in important_phrases.iterrows():
        tags[r["word"]] = r["score"]

    return tags



if __name__ == '__main__':


    print("Starting Preprocessing!")
    tsv_path = "../../data/politeness.tsv"
    output_path = "../../data/"

    #generate tags
    generate(source=tsv_path, dest=output_path, style1="P_0", style2="P_9")

    print("Finished Parallelizing")
