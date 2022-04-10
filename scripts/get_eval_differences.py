import ast
import csv
import json
import scipy.stats

QUESTIONS = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/retriever/qas/nq-test-ent.csv"
RETRIEVAL_DIR = "/n/fs/nlp-jacksond/projects/LUKE_DPR/outputs/retrieval_results/"
FIRST_RETRIEVAL = "luke_ent_nq_100_new.json"
SECOND_RETRIEVAL = "luke_ent_set_nq_100.json"

def main():
    q_ents = {}
    with open(QUESTIONS) as q_f:
        reader = csv.reader(q_f, delimiter="\t")
        for line in reader:
            try:
                q_ents[line[0]] = ast.literal_eval(line[2])
            except IndexError:
                continue
    
    with open(f"{RETRIEVAL_DIR}{FIRST_RETRIEVAL}") as first_f, open(f"{RETRIEVAL_DIR}{SECOND_RETRIEVAL}") as second_f:
        first_retrieval_results = json.load(first_f)
        second_retrieval_results = json.load(second_f)

    first_correct_qs = []
    second_correct_qs = []
    k = 20

    for first_results, second_results in zip(first_retrieval_results, second_retrieval_results):
        first_docs = first_results["ctxs"][:k]
        second_docs = second_results["ctxs"][:k]

        first_correct = False
        for doc in first_docs:
            if doc["has_answer"]:
                first_correct = True
                break

        second_correct = False
        for doc in second_docs:
            if doc["has_answer"]:
                second_correct = True
                break

        if first_correct == second_correct:
            continue

        if first_correct:
            first_correct_qs.append(first_results)
        else:
            second_correct_qs.append(second_results)

    issues = 0
    first_q_ents = []
    for question in first_correct_qs:
        try:
            ents = q_ents[question["question"]]
            first_q_ents.append(len(ents))
        except KeyError:
            issues += 1

    second_q_ents = []
    for question in second_correct_qs:
        try:
            ents = q_ents[question["question"]]
            second_q_ents.append(len(ents))
        except KeyError:
            issues += 1

    print("Question difference:")
    print(f"{FIRST_RETRIEVAL}: {len(first_correct_qs)}")
    print(f"{SECOND_RETRIEVAL}: {len(second_correct_qs)}")
    print()

    print("Average entities per question:")
    print(f"{FIRST_RETRIEVAL}: {sum(first_q_ents) / len(first_correct_qs)}")
    print(f"{SECOND_RETRIEVAL}: {sum(second_q_ents) / len(second_correct_qs)}")
    print(scipy.stats.mannwhitneyu(first_q_ents, second_q_ents))
    print()


if __name__ == "__main__":
    main()
