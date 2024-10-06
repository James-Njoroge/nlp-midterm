import pandas as pd
import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet
from collections import Counter
import nltk
import os
from hebrew_tokenizer import tokenize as hebrew_tokenize
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

def process_sentences(input_files, output_file, language):
    data = []
    pairid = 0
    contextid = 0
    sentid = 0
    
    if language == 'english':
        lemmatizer = WordNetLemmatizer()
    elif language in ['french', 'german', 'russian']:
        lemmatizer = SnowballStemmer(language)
    elif language == 'hebrew':
        lemmatizer = None  # Hebrew lemmatization not supported, will use tokens as is
    else:
        raise ValueError("Unsupported language. Supported languages are 'english', 'french', 'german', 'hebrew', 'russian'.")
    
    for input_file in input_files:
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
                if language == 'hebrew':
                    all_words = [token[1] for token in hebrew_tokenize(true_sentence)] + [token[1] for token in hebrew_tokenize(false_sentence)]
                else:
                    all_words = nltk.word_tokenize(true_sentence, language=language) + nltk.word_tokenize(false_sentence, language=language)
                word_freq = Counter(all_words)
                common_words = [word for word, freq in word_freq.items() if freq > 1]
                pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in common_words) + r')\b')
                
                # Find ROI
                roi = []
                for k in range(len(true_tokens)):
                    if true_tokens[k] != false_tokens[k]:
                        roi.append(str(k))
                roi_indices = [int(index) for index in roi] if roi else []
                
                for j, (sentence, boolean) in enumerate([(true_sentence, True), (false_sentence, False)]):
                    columns = {} # New row dictionary
                    columns['sentid'] = sentid + 1
                    columns['comparison'] = 'expected' if boolean else 'unexpected'
                    columns['pairid'] = pairid + 1
                    columns['contextid'] = contextid + 1
                    
                    # Lemmatize ROI word from both sentences in the pair
                    if roi_indices:
                        roi_word = true_tokens[roi_indices[0]] if boolean else false_tokens[roi_indices[0]]
                        if language == 'english':
                            pos_tag = nltk.pos_tag([roi_word])[0][1]
                            wordnet_pos = get_wordnet_pos(pos_tag)
                            columns['lemma'] = lemmatizer.lemmatize(roi_word, pos=wordnet_pos)
                        elif language in ['french', 'german', 'russian']:
                            columns['lemma'] = lemmatizer.stem(roi_word)
                        elif language == 'hebrew':
                            columns['lemma'] = roi_word  # No lemmatization for Hebrew, use the word as is
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
    "en_evalset/simple_agrmt.txt"
]
output_file = 'data/en_data/test_en.tsv'
language = 'english'  # Specify the language: 'en', 'fr', 'de', 'he', 'ru'
process_sentences(input_files, output_file, language)

print(f"Output saved to {output_file}")