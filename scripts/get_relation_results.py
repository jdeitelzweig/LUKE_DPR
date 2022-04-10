import json
import os
from collections import defaultdict

DATASET_DIR = "/n/fs/nlp-jacksond/datasets/entity-questions/test/"

# FIRST_RETRIEVAL_RESULTS = "/n/fs/nlp-jacksond/projects/LUKE_DPR/outputs/retrieval_results/luke_ent_eq_100_new.json"
FIRST_RETRIEVAL_RESULTS = "/n/fs/nlp-jacksond/projects/LUKE_DPR/outputs/retrieval_results/luke_changes_eq_100.json"
SECOND_RETRIEVAL_RESULTS = "/n/fs/nlp-jacksond/projects/LUKE_DPR/outputs/retrieval_results/bert_changes_eq_100.json"

def main():
    first_results_dict = defaultdict(bool)
    second_results_dict = defaultdict(bool)
    k = 20
    with open(FIRST_RETRIEVAL_RESULTS) as retrieval_f:
        results = json.load(retrieval_f)
        for result in results:
            for doc in result["ctxs"][:k]:
                if doc["has_answer"]:
                    first_results_dict[result["question"]] = True
                    break

    with open(SECOND_RETRIEVAL_RESULTS) as retrieval_f:
        results = json.load(retrieval_f)
        for result in results:
            for doc in result["ctxs"][:k]:
                if doc["has_answer"]:
                    second_results_dict[result["question"]] = True
                    break

    better_relations = 0
    total_relations = 0
    for file in sorted(os.listdir(DATASET_DIR)):
        with open(f"{DATASET_DIR}{file}") as f:
            relation_qs = json.load(f)
        
        first_correct = 0
        for q in relation_qs:
            if first_results_dict[q["question"]]:
                first_correct += 1
        second_correct = 0
        for q in relation_qs:
            if second_results_dict[q["question"]]:
                second_correct += 1
        
        print(f"{file} ({len(relation_qs)})):{100*(first_correct - second_correct)/len(relation_qs):.2f}")
        if first_correct > second_correct:
            better_relations += 1
        total_relations += 1

    print(f"Relations better: {better_relations}/{total_relations}")



if __name__ == "__main__":
    main()
