import argparse
import csv
import json
import os
from transformers import BertTokenizer

OUTPUT_PATH = "/n/fs/nlp-jacksond/projects/LUKE_DPR/data/"


def load_data(dataset_path, data_source, using_eq=False):
    if using_eq:
        data = []
        for eq_file in os.listdir(dataset_path + data_source):
            file_name = dataset_path + data_source + "/" + eq_file
            with open(file_name, "r") as f:
                questions = json.load(f)
                data.extend(questions)
        return data
    with open(dataset_path + data_source + "_preprocessed.json", "r") as f:
        return json.load(f)["data"]
            

def match_elq_output(data, elq_output_file):
    questions = []
    with open(elq_output_file) as f:
        btz = BertTokenizer.from_pretrained("bert-base-uncased")
        for question, output_line in zip(data, f):
            output_line = json.loads(output_line)
            assert output_line["text"].lower() == question["question"].lower()

            entity_ranges = []
            for ent, span in zip(output_line["pred_tuples_string"], output_line["pred_triples"]):
                # entity in text, real entity name, start token, end token
                entity_ranges.append((ent[1], ent[0], span[1], span[2]))

            token_pos = {}
            total_tokens = 0
            current_pos = 0
            for word in output_line["text"].split():
                word_token_ids = btz(word)['input_ids'][1:-1]
                for token_id in word_token_ids:
                    token = btz.decode([token_id])
                    if "##" in token:
                        token = token[2:]
                    token_pos[total_tokens] = current_pos
                    total_tokens += 1
                    current_pos += len(token)
                current_pos += word.count('\u200b')
                current_pos += 1 # for space
            token_pos[total_tokens] = current_pos - 1 # no space at end

            entity_spans = []

            for entity in entity_ranges:
                start = token_pos[entity[2]]
                end = token_pos[entity[3]]
                # get rid of space at end if exists
                if end < len(output_line["text"]) + 1 and output_line["text"][end-1] == ' ':
                    end -= 1
                entity_spans.append((entity[1], start, end))
            
            question_copy = question.copy()
            question_copy["ent_offsets"] = entity_spans
            questions.append(question_copy)
    return questions


def write_data(questions, output_name):
    with open(output_name, "w+") as f:
        question_writer = csv.writer(f, delimiter="\t")
        for question in questions:
            question_writer.writerow([question["question"], question["answers"], question["ent_offsets"]])


def main():
    parser = argparse.ArgumentParser(description='Add entity offsets to a QA dataset and convert to DPR format')
    parser.add_argument("--use_eq", action="store_true")
    parser.add_argument("--data_source", required=True)
    args = parser.parse_args()

    using_eq = args.use_eq
    data_source = args.data_source

    dataset_path = f"/n/fs/nlp-jacksond/datasets/{'entity-questions' if using_eq else 'nq-open'}/"
    elq_output_name = f"/n/fs/nlp-jacksond/projects/BLINK/output/{'eq' if using_eq else 'nq'}_{data_source}/biencoder_outs.jsonl"
    output_name = f"{OUTPUT_PATH}{'eq' if using_eq else 'nq'}-{data_source}-ent.csv"
    
    print(f"Loading data at {dataset_path}")
    data = load_data(dataset_path, data_source, using_eq)

    print(f"Matching with elq data at {elq_output_name}")
    questions = match_elq_output(data, elq_output_name)

    print(f"Writing to {output_name}")
    write_data(questions, output_name)


if __name__ == "__main__":
    main()
