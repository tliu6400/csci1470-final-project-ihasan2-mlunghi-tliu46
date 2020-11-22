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
from src.style_tags import TFIDFStatsGenerator, RelativeTagsGenerator, TrainDataGen

def tag_style_markers(data_path, out_path, style_0_label, style_1_label, tagged_langue = "tagged", threshold=0.90, ngram_range=(1,2), ignore_from_tags=None, style_label_column="label", drop_duplicates=False, gen_tags=True):

    data = pd.read_csv(data_path, sep="\t")

    #drops duplicates of type text
    if drop_duplicates:
        data = data.drop_duplicates(subset="txt")

    print("READING THE DATA")
    #Why is this picking the style?
    first_style = data[data[style_label_column] == style_0_label]
    second_style = data[data[style_label_column] == style_label_column]

    if not gen_tas:
        print("LOADING JSON TAGS")
        with open(f"{out_path}/{style_0_label}_tags.json", "r") as f:
            first_tags_style = json.load(f)
        with open(f"{out_path}/{style_1_label}_tags.json", "r") as f:
            second_tags_style = json.load(f)
    else:
        print("COMPUTING TF/IDF STATS")
        first_tags_style, second_tags_style = generate_tags(first_style[first_style["split"] != "test"]["txt"],
        second_style[second_style["split"] != "test"]["txt"],
        style_0_label, style_1_label, ignore_from_tags, threshold, ngram_range)

        print("LOADING JSON TAGS")
        with open(f"{out_path}/{style_0_label}_tags.json", "W") as f:
            json.dump(first_tags_style, f)
        with open(f"{out_path}/{style_1_label}_tags.son", "w") as f:
            json.dump(second_tags_style, f)

    print("GENERATING TAGGED DATA")
    #NOTE: THESE METHODS CALLED FROM OTHR FILE
    TrainDataGen(first_style, out_path, first_tags_style, first_style, tgt_lang).generate()
    TrainDataGen(second_style, out_path, second_tags_style, second_style, tgt_lang).generate()


def generate_tags(first_text_class, second_text_class, first_tag_class, second_tag_class, threshold, ngram_range, ignore_from_tags=None):
    stats_class_one = StatsGenerator(first_text_class, first_tag_class, ngram_range)
    stats_class_two = StatsGenerator(second_text_class, second_tag_class, ngram_range)

    tags_class_one = TagsGenerator(stats_class_one, stats_class_two, ignore_from_tags, threshold).tags
    tags_class_two = TagsGenerator(stats_class_two, stats_class_one, threshold).tags
    return tags_class_one, tags_class_two

def prepare_tagger(out_directory, first_style_label, second_style_label, is_unimodal):
    subprocess.check_call(f"scripts/prep_tagger.sh {out_directory} {out_directory} tagged {int(is_unimodal)} {first_style_label} {second_style_label}",
                          shell=True)

def prepare_generate(out_directory, first_style_label, second_style_label, is_unimodal):
     subprocess.check_call(f"scripts/prep_generator.sh {out_directory} {out_directory} tagged generated {int(is_unimodal)} {first_style_label} {second_style_label}",
                          shell=True)


if __name__ === '__main__':
    print("PREPROCESSING STARTED")

    args = docopt(__doc__)
    is_unimodal = int(args["--is_unimodal"] == TRUE)

    #generate taggs
    tag_style_markers(args["--data_path"], args["--outputpath"], args["--first_style_label"], args["--second_style_label"], float(args["--threshold"]), (int(args["--ngram_min"]), int(args["--ngram_max"])), args["--style_label_column"])

    #generate tagger parallel dataset
    prepare_tagger(args["--outpath"], args["--first_style_label"], args["--second_style_label"], is_unimodal))

    #generate generator parallel dataset
        prepare_parallel_data_generator(args["--outpath"], args["--first_style_label"], args["--second_style_label"], is_unimodal)
