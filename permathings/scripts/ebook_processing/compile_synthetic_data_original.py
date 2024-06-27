import os, sys, json, random, string, re, time

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

SCS = SWIFT_CONTEXT_START = "<|swift_context_start|>"
SCE = SWIFT_CONTEXT_END = "<|swift_context_end|>"
NCS = NORMAL_CONTEXT_START = "<|nonswift_context_start|>"
NCE = NORMAL_CONTEXT_END = "<|nonswift_context_end|>"
PTSS = PART_TO_BE_SWIFTIFIED_START = "<|to_swiftify_start|>"
PTSE = PART_TO_BE_SWIFTIFIED_END = "<|to_swiftify_end|>"
SNS = SWIFTIFICATION_START = "<|swift_start|>"
SNE = SWIFTIFICATION_END = "<|swift_end|>"

"""
patterns:
SCS + swift_precontext + SCE + NCS + nonswift_precontext + NCE + PTSS + part_to_be_swiftified + PTSE + SNS + swiftified + SNE
NCS + nonswift_precontext + NCE + PTSS + part_to_be_preswiftified + PTSE + SNS + preswiftified + SNE
SCS + swift_context + SCE + NCS + nonswift_context + NCE + PTSS + part_to_be_swiftified + PTSE + SCS + swift_context + SCE + SNS + swiftified + SNE
"""

final_dataset = []

for chunk_filename in all_chunks_data:
    chunk_data = all_chunks_data[chunk_filename]
    sentences_swift = [sentence_pair["swift"] for sentence_pair in chunk_data["sentence_pairs"]]
    sentences_plain = [sentence_pair["plain"] for sentence_pair in chunk_data["sentence_pairs"]]
    for sentence_pair_num in range(1,len(chunk_data["sentence_pairs"])-1):
        prior_swift = " ".join(chunk_data["given_info_as_json_dict"]["prior_context"].split(" ")[2+random.randint(0,10):]).strip()
        post_swift = " ".join(chunk_data["given_info_as_json_dict"]["post_context"].split(" ")[:-random.randint(0,10)-2]).strip()
        prior_nonswift = " ".join(sentences_plain[:sentence_pair_num]).strip()
        post_nonswift = " ".join(sentences_plain[sentence_pair_num+1:]).strip()
        nonswift_sentence = sentences_plain[sentence_pair_num]
        swift_sentence = sentences_swift[sentence_pair_num]
        if len(prior_swift) and len(prior_nonswift):
            prior_block1 = SWIFT_CONTEXT_START + " " + prior_swift + " " + SWIFT_CONTEXT_END + " " + NORMAL_CONTEXT_START + " " + prior_nonswift + " " + NORMAL_CONTEXT_END
            prior_block2 = NORMAL_CONTEXT_START + " " + prior_nonswift + " " + NORMAL_CONTEXT_END
        elif len(prior_nonswift):
            prior_block1 = prior_block2 = NORMAL_CONTEXT_START + " " + prior_nonswift + " " + NORMAL_CONTEXT_END
        else:
            prior_block1 = ""
        if len(post_nonswift):
            post_block = NORMAL_CONTEXT_START + " " + post_nonswift + " " + NORMAL_CONTEXT_END + " " + SWIFT_CONTEXT_START + " " + post_swift + " " + SWIFT_CONTEXT_END
        elif len(post_swift):
            post_block = SWIFT_CONTEXT_START + " " + post_swift + " " + SWIFT_CONTEXT_END
        else:
            post_block = ""
        sentence_block = PART_TO_BE_SWIFTIFIED_START + " " + nonswift_sentence + " " + PART_TO_BE_SWIFTIFIED_END
        full_block = prior_block1 + " " + sentence_block + " " + post_block + " " + SWIFTIFICATION_START + " " + swift_sentence + " " + SWIFTIFICATION_END
        final_dataset.append(full_block)
        full_block = prior_block1 + " " + sentence_block + " " + SWIFTIFICATION_START + " " + swift_sentence + " " + SWIFTIFICATION_END
        final_dataset.append(full_block)
        if prior_block1 != prior_block2:
            full_block = prior_block2 + " " + sentence_block + " " + post_block + " " + SWIFTIFICATION_START + " " + swift_sentence + " " + SWIFTIFICATION_END
            final_dataset.append(full_block)
            full_block = prior_block2 + " " + sentence_block + " " + SWIFTIFICATION_START + " " + swift_sentence + " " + SWIFTIFICATION_END
            final_dataset.append(full_block)
            

random.shuffle(final_dataset)
#save the dataset
with open(os.path.join(ebook_analyses_dir, "final_dataset.json"), "w") as f:
    json.dump(final_dataset, f, indent=4)