import pandas as pd
import re
from collections import Counter
import os
from transformers import BertTokenizer, BertModel
import torch

def process_sentences(input_files, output_files, language):
    # Load BERT tokenizer and model
    if language == 'english':
        model_name = 'bert-base-uncased'
    else:
        model_name = 'bert-base-multilingual-cased'
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    for input_file, output_file in zip(input_files, output_files):
        data = []
        pairid = 0
        contextid = 0
        sentid = 0
        
        condition = os.path.splitext(os.path.basename(input_file))[0]
        condition = condition.replace('_anim', '').replace('_inanim', '')
        
        with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 >= len(lines):
                    # Skip if there is no corresponding false sentence
                    break
                true_sentence_parts = lines[i].strip().split('\t')
                false_sentence_parts = lines[i + 1].strip().split('\t')
                
                if len(true_sentence_parts) < 2 or len(false_sentence_parts) < 2:
                    # Skip if the line format is incorrect
                    continue
                
                true_sentence = true_sentence_parts[-1]
                false_sentence = false_sentence_parts[-1]
                
                # Ensure both sentences have the same number of tokens
                true_tokens = true_sentence.split()
                false_tokens = false_sentence.split()
                if len(true_tokens) != len(false_tokens):
                    # Skip if the number of tokens does not match
                    continue
                
                # Extract common words dynamically from both true and false sentences
                all_words = tokenizer.tokenize(true_sentence) + tokenizer.tokenize(false_sentence)
                word_freq = Counter(all_words)
                common_words = [word for word, freq in word_freq.items() if freq > 1]
                pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in common_words) + r')\b')
                
                # Find ROI
                roi = []
                for k in range(len(true_tokens)):
                    if true_tokens[k] != false_tokens[k]:
                        roi.append(str(k))
                roi_indices = [int(index) for index in roi] if roi else []
                
                # Set lemma for the pair
                lemma = true_tokens[roi_indices[0]] if roi_indices else ''
                
                for j, (sentence, boolean) in enumerate([(true_sentence, True), (false_sentence, False)]):
                    columns = {} # New row dictionary
                    columns['sentid'] = sentid + 1
                    columns['comparison'] = 'expected' if boolean else 'unexpected'
                    columns['pairid'] = pairid + 1
                    columns['contextid'] = contextid + 1
                    
                    # Use the same lemma for both expected and unexpected conditions
                    columns['lemma'] = lemma
                    
                    columns['condition'] = condition
                    columns['sentence'] = sentence
                    columns['ROI'] = ','.join(str(int(index)) for index in roi)
                    
                    data.append(columns)
                    sentid += 1
                
                pairid += 1
                contextid += 1

        df = pd.DataFrame(data, columns=[
            'sentid', 'comparison', 'pairid', 'contextid', 'lemma', 'condition', 'sentence', 'ROI'
        ])
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Output saved to {output_file}")

input_files = [
    "en_evalset/simple_agrmt.txt",
    "en_evalset/vp_coord.txt",
    "en_evalset/long_vp_coord.txt",
    "en_evalset/subj_rel.txt",
    "en_evalset/obj_rel_within_anim.txt",
    "en_evalset/obj_rel_across_anim.txt",
    "en_evalset/prep_anim.txt", # cutoff for non-english langs
    "en_evalset/obj_rel_across_inanim.txt",
    "en_evalset/obj_rel_no_comp_across_anim.txt",
    "en_evalset/obj_rel_no_comp_across_inanim.txt",
    "en_evalset/obj_rel_no_comp_within_anim.txt",
    "en_evalset/obj_rel_no_comp_within_inanim.txt",
    "en_evalset/obj_rel_within_inanim.txt",
    "en_evalset/prep_inanim.txt",
    "en_evalset/reflexive_sent_comp.txt",
    "en_evalset/reflexives_across.txt",
    "en_evalset/sent_comp.txt",
    "en_evalset/simple_reflexives.txt"
]
output_files = [
    "data/en_data/simple_agrmt_en.tsv",
    "data/en_data/vp_coord_en.tsv",
    "data/en_data/long_vp_coord_en.tsv",
    "data/en_data/subj_rel_en.tsv",
    "data/en_data/obj_rel_within_anim_en.tsv",
    "data/en_data/obj_rel_across_anim_en.tsv",
    "data/en_data/prep_anim_en.tsv", # cutoff for non-english langs
    "data/en_data/obj_rel_across_inanim_en.tsv",
    "data/en_data/obj_rel_no_comp_across_anim_en.tsv",
    "data/en_data/obj_rel_no_comp_across_inanim_en.tsv",
    "data/en_data/obj_rel_no_comp_within_anim_en.tsv",
    "data/en_data/obj_rel_no_comp_within_inanim_en.tsv",
    "data/en_data/obj_rel_within_inanim_en.tsv",
    "data/en_data/prep_inanim_en.tsv",
    "data/en_data/reflexive_sent_comp_en.tsv",
    "data/en_data/reflexives_across_en.tsv",
    "data/en_data/sent_comp_en.tsv",
    "data/en_data/simple_reflexives_en.tsv"
]
language = 'english'  # Specify the language: 'en', 'fr', 'de', 'he', 'ru'
process_sentences(input_files, output_files, language)