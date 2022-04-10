import ast
import csv
import json
import os

LUKE_ENTITIES = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/luke_entity_vocab.json"
DATASET_ENTITIES = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/eq_entity_list.json"
DATASET = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/retriever/qas/eq-test-ent.csv"
DATASET_DIR = "/n/fs/nlp-jacksond/datasets/entity-questions/test/"
FILTER_LIST = ["Capital city", "Country", "Language", "Record label", "The Who"]

def get_dataset_entities(dataset):
    entities = {}
    with open(dataset) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            try:
                q_ents = ast.literal_eval(line[2])
                q_ents = [ent[0] for ent in q_ents]
                entities[line[0]] = q_ents
            except IndexError:
                continue
    return entities

def main():
    with open(LUKE_ENTITIES) as luke_f, open(DATASET_ENTITIES) as data_f:
        luke_ents = json.load(luke_f).keys()
        data_ents = json.load(data_f)

        found = 0
        total = 0
        for ent in data_ents:
            if ent in FILTER_LIST:
                continue

            if ent in luke_ents:
                found += 1
            total += 1
        print(found / total)

def main2():
    entities_map = get_dataset_entities(DATASET)
    with open(LUKE_ENTITIES) as luke_f:
        luke_ents = json.load(luke_f).keys()

    for file in sorted(os.listdir(DATASET_DIR)):
        with open(f"{DATASET_DIR}{file}") as f:
            relation_qs = json.load(f)

        total = 0
        overlap = 0
        for q in relation_qs:
            ents = entities_map[q["question"]]
            for ent in ents:
                if ent in FILTER_LIST:
                    continue

                if ent in luke_ents:
                    overlap += 1
                total += 1
        print(f"{file}: {100*overlap / total:.2f}")


if __name__ == "__main__":
    main2()