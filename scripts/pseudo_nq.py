import argparse
import ast
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.summarization.bm25 import BM25

WIKI_DATA = "/n/fs/nlp-jacksond/projects/EFDPR/data/wikipedia_splits/psgs_w100_ent.tsv"

# initialize tokenizer
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

@dataclass(frozen=True)
class Passage:
    passage_id: int
    text: str
    title: str
    entities: list = field(hash=False)

    def contains_answer(self, answers) -> bool:
        for answer in answers:
            if answer in self.text:
                return True
        return False

def tokenize(text):
    return [t.text for t in tokenizer(text)]


def get_sorted_docs(query, docs):
    if not docs:
        return []

    texts = [tokenize(doc) for doc in docs]

    bm25_obj = BM25(texts)
    scores = bm25_obj.get_scores(tokenize(query))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def get_new_passages(original_passages, wiki_data, used_ids, answers, should_contain_answer, use_first=False):
    used_passages = []
    for original_passage in original_passages:
        title = original_passage["title"]

        possible_passages = wiki_data[title]

        if not use_first:
            doc_iterator = get_sorted_docs(original_passage["text"], [passage.text for passage in possible_passages])
        else:
            doc_iterator = range(len(possible_passages))

        found_passage = False
        for i in doc_iterator:
            poss_passage = possible_passages[i]
            if poss_passage.passage_id not in used_ids and (should_contain_answer == poss_passage.contains_answer(answers)):
                # make sure we don't use this passage again
                used_ids.add(poss_passage.passage_id)
                used_passages.append(asdict(poss_passage))
                found_passage = True
                break

        # if no valid candidates, append original with no entitiy annotations
        if not found_passage:
            used_passages.append(original_passage)

    return used_passages, used_ids


def main():
    parser = argparse.ArgumentParser(description='Add entity offsets to a QA dataset and convert to DPR format')
    parser.add_argument("--nq_data", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--use_first", action="store_true")
    args = parser.parse_args()

    # dictionary of all passages from wiki data by title
    passages = defaultdict(list)

    print("Reading wiki data")
    with open(WIKI_DATA) as wiki_file:
        wiki_reader = csv.reader(wiki_file, delimiter='\t')
        # skip headers
        next(wiki_reader)

        invalid_lines = []
        for line in wiki_reader:
            try:
                title = line[2]
                text = line[1]

                if not text:
                    invalid_lines.append(line)
                    continue
                passages[title].append(Passage(line[0], line[1], title, line[3]))#ast.literal_eval(line[3])))
            except IndexError:
                invalid_lines.append(line)
        print(f"Invalid lines: {len(invalid_lines)}")

    print("Reading nq data")
    with open(args.nq_data) as nq_file:
        nq_data = json.load(nq_file)
        
        print("Processing nq data")
        for question in tqdm(nq_data):
            answers = question["answers"]
            used_passage_ids = set()
            used_positive_passages, used_passage_ids = get_new_passages(question["positive_ctxs"], passages, used_passage_ids, answers, True, use_first=args.use_first)            
            used_negative_passages, used_passage_ids = get_new_passages(question["negative_ctxs"], passages, used_passage_ids, answers, False, use_first=args.use_first)
            used_hard_negative_passages, used_passage_ids = get_new_passages(question["hard_negative_ctxs"], passages, used_passage_ids, answers, False, use_first=args.use_first)
            
            question["positive_ctxs"] = used_positive_passages
            question["negative_ctxs"] = used_negative_passages
            question["hard_negative_ctxs"] = used_hard_negative_passages
        
        with open(args.output_file, "w+") as out_file:
            json.dump(nq_data, out_file, indent=4)


if __name__ == "__main__":
    main()