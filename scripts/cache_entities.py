import ast
import csv
import json

CSV_FILE = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/wikipedia_splits/psgs_w100_ent.tsv"

passages_dict = {}

with open(CSV_FILE) as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)

    for row in reader:
        passage_id = row[0]
        try:
            ent_offsets = ast.literal_eval(row[3])
        except IndexError:
            continue
        passages_dict[passage_id] = ent_offsets

with open("cached_entities.json", "w+") as f:
    json.dump(passages_dict, f, indent=4)
