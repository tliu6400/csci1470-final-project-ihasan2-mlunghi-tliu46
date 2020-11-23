"""Generates tags
Usage:
    preprocess.py [options]

Options:
    --data_path=<str>                        Path to the data directory
    --outpath=<str>                         Output path
    --first_style_label=<str>                   Label for style 0
    --second_style_label=<str>                   Label for style 1
    --ngram_min=<int>                 Min n_gram_range [default: 1]
    --ngram_max=<int>                 Max n_gram_range [default: 2]
    --style_label_column=<str>                 Name of the column that has style label column [default: style]
    --threshold=<float>                        tf-idf ratio threshold [default: 0.90]
    --is_unimodal=<bool>                    Whether the dataset is unimodal (like politeness) or has two styles (like yelp)
"""
from docopt import docopt
import json
import pandas as pd
import pandas as pd
import numpy as np
import tempfile
import sys
import subprocess
from collections import Counter
from typing import List
import logging

#NEED TO CHANGE FUNCTION NAMES
from original_styletags import TFIDFStatsGenerator, RelativeTagsGenerator, TrainDataGen

def tag_style_markers(data_path, out_path, style_0_label, style_1_label, ngram_range, tagged_lang = "tagged", threshold=0.90, ignore_from_tags=None, style_label_column=None, drop_duplicates=False, gen_tags=True):

    data = pd.read_csv(data_path, sep="\t")

    #drops duplicates of type text
    if drop_duplicates:
        data = data.drop_duplicates(subset="txt")

    print("READING THE DATA")
    #Why is this picking the style?
    first_style = data[data[style_label_column] == style_0_label]
    second_style = data[data[style_label_column] == style_1_label]

    if not gen_tags:
        print("LOADING JSON TAGS")
        with open(f"{out_path}/{style_0_label}_tags.json", "r") as f:
            first_tags_style = json.load(f)
        with open(f"{out_path}/{style_1_label}_tags.json", "r") as f:
            second_tags_style = json.load(f)
    else:
        print("COMPUTING TF/IDF STATS")
        first_tags_style, second_tags_style = generate_tags(first_text_class=first_style[first_style["split"] != "test"]["txt"],second_text_class=second_style[second_style["split"] != "test"]["txt"], first_tag_class=style_0_label, second_tag_class=style_1_label, ignore_from_tags=None, threshold=threshold, ngram_range=ngram_range)

        print("LOADING JSON TAGS")
        with open(f"{out_path}/{style_0_label}_tags.json", "w") as f:
            json.dump(first_tags_style, f)
        with open(f"{out_path}/{style_1_label}_tags.json", "w") as f:
            json.dump(second_tags_style, f)

    print("GENERATING TAGGED DATA")
    #NOTE: THESE METHODS CALLED FROM OTHR FILE
    TrainDataGen(first_style, out_path, first_tags_style, first_style, tagged_lang).generate()
    TrainDataGen(second_style, out_path, second_tags_style, second_style, tagged_lang).generate()


def generate_tags(first_text_class, second_text_class, first_tag_class, second_tag_class, threshold, ngram_range, ignore_from_tags=None):
    #print(first_text_class)
    #print(first_tag_class)
    #print(ngram_range)
    stats_class_one = TFIDFStatsGenerator(data=first_text_class, data_id=first_tag_class, ngram_range=ngram_range)
    #print(second_text_class)
    #print(second_tag_class)
    #print(ngram_range)
    stats_class_two = TFIDFStatsGenerator(data=second_text_class, data_id=second_tag_class, ngram_range=ngram_range)

    tags_class_one = RelativeTagsGenerator(main_class_stats=stats_class_one, relative_class_stats=stats_class_two, ignore_from_tags=ignore_from_tags, thresh=threshold).tags
    tags_class_two = RelativeTagsGenerator(main_class_stats=stats_class_two, relative_class_stats=stats_class_one, thresh=threshold).tags
    return tags_class_one, tags_class_two

def prepare_tagger(out_directory, first_style_label, second_style_label, is_unimodal):
    subprocess.check_call(f"scripts/prep_tagger.sh {out_directory} {out_directory} tagged {int(is_unimodal)} {first_style_label} {second_style_label}",
                          shell=True)

def prepare_generate(out_directory, first_style_label, second_style_label, is_unimodal):
     subprocess.check_call(f"scripts/prep_generator.sh {out_directory} {out_directory} tagged generated {int(is_unimodal)} {first_style_label} {second_style_label}",
                          shell=True)


if __name__ == '__main__':
    print("PREPROCESSING STARTED")

    args = docopt(__doc__, version="")
    is_unimodal = int(args["--is_unimodal"] == True)

    #generate taggs
    tag_style_markers(data_path=args["--data_path"], out_path=args["--outpath"], style_0_label=args["--first_style_label"], style_1_label=args["--second_style_label"], threshold=float(args["--threshold"]), ngram_range=(int(args["--ngram_min"]), int(args["--ngram_max"])), style_label_column=args["--style_label_column"])


    #generate tagger parallel dataset
    prepare_tagger(args["--outpath"], args["--first_style_label"], args["--second_style_label"], is_unimodal)

    #generate generator parallel dataset
    prepare_parallel_data_generator(args["--outpath"], args["--first_style_label"], args["--second_style_label"], is_unimodal)

    print("PARALLELIZING DONE")
