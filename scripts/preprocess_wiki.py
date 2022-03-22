import bz2
import json
import os
import re
import urllib.parse
from tqdm import tqdm

HOTPOT_DIR = "/n/fs/nlp-jacksond/datasets/hotpot-wiki/"
OUTPUT_FILE = "/n/fs/nlp-jacksond/projects/EFDPR/data/psgs_w100_ent.tsv"
HTML_TAG_REGEX = re.compile(r'<a[^<>]href=([\'\"])(.*?)\1>(.*?)<\/a>', re.IGNORECASE)

passage_id = 1
with open(OUTPUT_FILE, "w+") as out_file:
    # Write headers
    out_file.write("\t".join(["id", "text", "title", "ent_offsets"]) + "\n")

    for sub_dir in tqdm(sorted(os.listdir(HOTPOT_DIR))):
        for file in sorted(os.listdir(HOTPOT_DIR + sub_dir)):
            with bz2.open(HOTPOT_DIR + sub_dir + "/" + file) as f:
                data = f.read().decode()
                data = data.split('\n')

                if (data[-1] == ''):
                    data = data[:-1]

                for line in data:
                    # convert line to json
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        print(line)
                        continue

                    # turn paragraphs into one block of text
                    full_text = ""
                    for para in article["text"][1:]:
                        full_text += "".join(para)

                    # find all entity mentions and replace the <a> tags with the inner text
                    ent_mentions = []

                    match = re.search(HTML_TAG_REGEX, full_text)
                    end = 0
                    while match:
                        end += match.start() + len(match.group(3))
                        ent_name = urllib.parse.unquote(match.group(2))
                        ent_mentions.append((ent_name, end - len(match.group(3)), end))
                        full_text = re.sub(HTML_TAG_REGEX, r'\3', full_text, 1)
                        match = re.search(HTML_TAG_REGEX, full_text[end:])

                    # split text into words
                    words = full_text.split(" ")

                    # split article into 100 word passages
                    # find entity menntions within article
                    # if entity is on bounndary, don't include it as an entity
                    i = 0
                    start_ind = 0
                    cur_mention = 0
                    while i < len(words):
                        cur_words = words[i:i+100]
                        cur_text = " ".join(cur_words)
                        # exclusive end index
                        end_ind = start_ind + len(cur_text)
                        cur_mentions = []

                        # looping in this fashion should take care of entities on boundaries
                        while cur_mention < len(ent_mentions) and ent_mentions[cur_mention][2] < end_ind:
                            if start_ind <= ent_mentions[cur_mention][1]:
                                # Subtract i / 100 to account for spaces between passages
                                new_start = ent_mentions[cur_mention][1] - start_ind - i // 100
                                new_end = ent_mentions[cur_mention][2] - start_ind - i // 100
                                cur_mentions.append((ent_mentions[cur_mention][0], new_start, new_end))
                            cur_mention += 1

                        # Writing this way instead of with csv.writer because of weird behavior with quotechars
                        out_file.write("\t".join([str(passage_id), cur_text, article["title"], str(cur_mentions)]) + "\n")
                        
                        passage_id += 1
                        i += 100
                        start_ind = end_ind