import os, sys, json, random, string, re, time
import itertools

this_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(this_dir)
permathings_dir = os.path.dirname(scripts_dir)
root_dir = os.path.dirname(permathings_dir)
ephemeral_dir = os.path.join(root_dir, "ephemera")
outputs_dir = os.path.join(ephemeral_dir, "outputs")
ebook_analyses_dir = os.path.join(outputs_dir, "ebook_analyses")
swift_quotes_dir = os.path.join(ebook_analyses_dir, "swift_quotes")

REPLACEMENTS = {
    "\u2019": "'",
    "\u2014": "-",
    "\u201c": "\"",
    "\u201d": "\"",
    "\u2026": "...",
    "\u2013": "-",
    "\u00a0": " ",
}

SKIP_STRINGS=[
    "_",
    "normalized",
    "\\n",
    "Z - nds",
    "G -- ",
    "d - l",
    "This quote",
    "----",
    "si- lent",
]

SKIP_IF_NOT_EQUALLY_IN_BOTH = [
    "---",
    "[",
    "]",
    "...",
    "#",
    ".--"
]

REMOVE_IF_STARTS_WITH = [
    ", ",
    "-- ",
    ": ",
    ". ",
]

all_chunks_data = {}
all_sentence_pairs = {}
for file in os.listdir(swift_quotes_dir):
    if file.endswith(".json"):
        with open(os.path.join(swift_quotes_dir, file), "r") as f:
            chunk_data = json.load(f)
        input_sentences = list(chunk_data["given_info_as_json_dict"]["chunk_sentences_to_normalize"].values())
        output_sentences = list(chunk_data["output"].values())
        sentence_pairs = []
        skip_this_one = False
        for i in range(len(input_sentences)):
            sentence_pair = {
                "swift": input_sentences[i],
                "plain": output_sentences[i]
            }
            if sentence_pair["swift"].lower() in sentence_pair["plain"].lower() or sentence_pair["plain"].lower() in sentence_pair["swift"].lower():
                skip_this_one = True
                break
            for replacement in REPLACEMENTS:
                sentence_pair["swift"] = sentence_pair["swift"].replace(replacement, REPLACEMENTS[replacement])
                sentence_pair["plain"] = sentence_pair["plain"].replace(replacement, REPLACEMENTS[replacement])
            if sentence_pair["swift"].split(" ")[0].replace(".", "").replace(":","").isnumeric():
                if not sentence_pair["plain"].split(" ")[0].replace(".", "").replace(":","").isnumeric():
                    sentence_pair["swift"] = " ".join(sentence_pair["swift"].split(" ")[1:])
            for skip_string in SKIP_STRINGS:
                if skip_string in sentence_pair["swift"] or skip_string in sentence_pair["plain"]:
                    skip_this_one = True
                    break
            if sentence_pair["swift"].startswith("\"") and sentence_pair["swift"].endswith("\""):
                if not sentence_pair["plain"].startswith("\"") and not sentence_pair["plain"].endswith("\""):
                    sentence_pair["swift"] = sentence_pair["swift"][1:-1]
            for skip_in_both_string in SKIP_IF_NOT_EQUALLY_IN_BOTH:
                if sentence_pair["swift"].count(skip_in_both_string) != sentence_pair["plain"].count(skip_in_both_string):
                    skip_this_one = True
                    break
            if sentence_pair["swift"].endswith("Mons."):
                skip_this_one = True
                break
            if sentence_pair["swift"].endswith(" .") and not sentence_pair["swift"].endswith(". ."):
                sentence_pair["swift"] = sentence_pair["swift"][:-2]+"."
            if sentence_pair["swift"].count("\"") % 2 != 0:
                skip_this_one = True
                break
            if sentence_pair["swift"].count("\"") != sentence_pair["plain"].count("\"") or \
                sentence_pair["swift"].count("(") != sentence_pair["plain"].count("(") or \
                sentence_pair["swift"].count(")") != sentence_pair["plain"].count(")"):
                skip_this_one = True
                break
            if sentence_pair["swift"].endswith("..") and not sentence_pair["swift"].endswith("..."):
                sentence_pair["swift"] = sentence_pair["swift"][:-1]
            if skip_this_one:
                break
            if sentence_pair["swift"].endswith("?") and not sentence_pair["plain"].endswith("?"):
                skip_this_one = True
                break
            if sentence_pair["swift"][0].islower() or sentence_pair["plain"][0].islower():
                skip_this_one = True
                break
            #if the first word of the swift sentence is all caps and length >1, make all but the first letter lowercase
            if sentence_pair["swift"].split(" ")[0].isupper() and len(sentence_pair["swift"].split(" ")[0]) > 1:
                sentence_pair["swift"] = sentence_pair["swift"].split(" ")[0][0] + sentence_pair["swift"].split(" ")[0][1:].lower() + " " + " ".join(sentence_pair["swift"].split(" ")[1:])
            for word in sentence_pair["swift"].split(" "):
                word = word.strip().replace(".","").replace(",","")
                if len(word) <2:
                    continue
                if word[-1].isnumeric() and not word[:-2].isnumeric():
                    skip_this_one = True
                    break
                if len(word) <3:
                    continue
                if word[-2].isnumeric() and not word[:-3].isnumeric():
                    skip_this_one = True
                    break
            for remove_string in REMOVE_IF_STARTS_WITH:
                if sentence_pair["swift"].startswith(remove_string):
                    sentence_pair["swift"] = sentence_pair["swift"][len(remove_string):]
                if sentence_pair["plain"].startswith(remove_string):
                    sentence_pair["plain"] = sentence_pair["plain"][len(remove_string):]
            sentence_pairs.append(sentence_pair)
        if skip_this_one:
            continue
        all_sentence_pairs[file] = sentence_pairs
        all_chunks_data[file] = chunk_data.copy()
        all_chunks_data[file]["sentence_pairs"] = sentence_pairs.copy()

#sort by key
all_sentence_pairs = dict(sorted(all_sentence_pairs.items()))

modernized_list = []
for chunk_filename in all_sentence_pairs:
    this_chunk_data = []
    for sentence_pair in all_sentence_pairs[chunk_filename]:
        this_chunk_data.append({
            "original": sentence_pair["swift"],
            "modernized": sentence_pair["plain"]
        })
    modernized_list.append(this_chunk_data)

with open(os.path.join(ebook_analyses_dir, "all_chunks_data.json"), "w") as f:
    json.dump(modernized_list, f, indent=4)

######################

SPECIAL_TOKENS={
    "CONTEXT": "<|context|>",
    "TO_CONVERT": "<|plain|>",
    "CONVERTED": "<|swiftify|>"
}

final_dataset = []

all_lines = []

for chunk_filename in all_chunks_data:
    chunk_data = all_chunks_data[chunk_filename]
    sentences_swift = [sentence_pair["swift"] for sentence_pair in chunk_data["sentence_pairs"]]
    sentences_plain = [sentence_pair["plain"] for sentence_pair in chunk_data["sentence_pairs"]]
    if len(sentences_swift) != len(sentences_plain):
        print("ERROR: Mismatch in number of sentences")
        input()
        continue
    start_from_sentence = max(1,len(sentences_swift)-10)
    for i in range(start_from_sentence, len(sentences_swift)):
        this_sentence = sentences_plain[i]
        this_context = sentences_plain[:i]
        converted_to = sentences_swift[i]
        if this_sentence == converted_to:
            continue
        if len(this_sentence) < 10 or len(converted_to) < 10:
            continue
        this_line = SPECIAL_TOKENS["CONTEXT"] + " " + " ".join(this_context) + " " + SPECIAL_TOKENS["TO_CONVERT"] + " " + this_sentence + " " + SPECIAL_TOKENS["CONVERTED"] + " " + converted_to
        all_lines.append(this_line)

for datum in all_lines:
    sample_of_other_lines = [datum,] + random.sample(all_lines, 10)
    full_string = (" ").join(sample_of_other_lines)
    final_dataset.append(full_string)

random.shuffle(final_dataset)
#save the dataset
with open(os.path.join(ebook_analyses_dir, "final_dataset.json"), "w") as f:
    json.dump(final_dataset, f, indent=4)