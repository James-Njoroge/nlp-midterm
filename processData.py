import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import nltk
import os
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def process_sentences(input_files, output_file):
    data = []
    pairid = 0
    contextid = 0
    sentid = 0
    lemmatizer = WordNetLemmatizer()
    
    for input_file in input_files:
        condition = os.path.splitext(os.path.basename(input_file))[0]
        condition = condition.replace('_anim', '').replace('_inanim', '')
        
        with open(input_file, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                true_sentence = lines[i].strip().split('\t')[1]
                false_sentence = lines[i+1].strip().split('\t')[1]
                
                # Extract common words dynamically from both true and false sentences
                all_words = nltk.word_tokenize(true_sentence) + nltk.word_tokenize(false_sentence)
                word_freq = Counter(all_words)
                common_words = [word for word, freq in word_freq.items() if freq > 1]
                pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in common_words) + r')\b')
                
                # Find ROI
                true_tokens = true_sentence.split()
                false_tokens = false_sentence.split()
                roi = []
                for k in range(len(true_tokens)):
                    if true_tokens[k] != false_tokens[k]:
                        roi.append(str(k))
                roi_indices = [int(index) - 1 for index in roi] if roi else []
                
                for j, (sentence, boolean) in enumerate([(true_sentence, True), (false_sentence, False)]):
                    columns = {} # New row dictionary
                    columns['sentid'] = sentid + 1
                    columns['comparison'] = 'expected' if boolean else 'unexpected'
                    columns['pairid'] = pairid + 1
                    columns['contextid'] = contextid + 1
                    
                    # Lemmatize ROI word from both sentences in the pair
                    if roi_indices:
                        roi_word = true_tokens[roi_indices[0]] if boolean else false_tokens[roi_indices[0]]
                        pos_tag = nltk.pos_tag([roi_word])[0][1]
                        wordnet_pos = get_wordnet_pos(pos_tag)
                        columns['lemma'] = lemmatizer.lemmatize(roi_word, pos=wordnet_pos)
                    else:
                        columns['lemma'] = ''
                    
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

input_files = [
    "en_evalset/simple_agrmt.txt",
    "en_evalset/sent_comp.txt",
    "en_evalset/vp_coord.txt",
    "en_evalset/long_vp_coord.txt",
    "en_evalset/subj_rel.txt",
    "en_evalset/obj_rel_within_anim.txt",
    "en_evalset/obj_rel_within_inanim.txt",
    "en_evalset/obj_rel_no_comp_within_anim.txt",
    "en_evalset/obj_rel_no_comp_within_inanim.txt",
    "en_evalset/obj_rel_across_anim.txt",
    "en_evalset/obj_rel_across_inanim.txt",
    "en_evalset/obj_rel_no_comp_across_anim.txt",
    "en_evalset/obj_rel_no_comp_across_inanim.txt",
    "en_evalset/prep_anim.txt",
    "en_evalset/prep_inanim.txt",
    "en_evalset/simple_reflexives.txt",
    "en_evalset/reflexive_sent_comp.txt",
    "en_evalset/reflexives_across.txt"
]
output_file = 'data/en_data/mono_en.tsv'
process_sentences(input_files, output_file)

print(f"Output saved to {output_file}")