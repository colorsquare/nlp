#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown, stopwords
from collections import defaultdict
import random, pprint

# ============================================================== #
# ====================== Global Variables ====================== #
# ============================================================== #

# Brown text corpora
stopwords_en = stopwords.words('english')
tagged_words = brown.tagged_words(tagset="universal")
tagged_bigrams = [
    ((word1.lower(), tag1), (word2.lower(), tag2))
    for (word1, tag1), (word2, tag2) in nltk.bigrams(tagged_words)
    # Filter out punctuations and bigrams with it.
    if not (tag1 == '.' or tag2 == '.')
]

# NLTK frequency distribution.
freqDist = nltk.FreqDist(tagged_bigrams)
conditionalFreqDist = nltk.ConditionalFreqDist(tagged_bigrams)
oppositeConditionalFreqDist = nltk.ConditionalFreqDist(
    (back, front) for front, back in tagged_bigrams
)

# Intensifiers from NLTK wordnet. 
def get_intensifiers():
    """
    Process intensity-modifying words, from NLTK wordnet corpus. 
    Returns list of intensifiers with universal tag.
    [
        (intensifier, universal_tag), ...
    ]
    """
    # keywords that are likely to be in intensity-modifying words. 
    keywords = ['extent', 'intensifier', 'intensity', 'quantifier', 'degree', 'comparative']
    intensifiers = []
    # Mapping dictionary from synset pos to universal tag.
    wnpos_to_universal = {
        'a': 'ADJ',
        's': 'ADJ',
        'r': 'ADV',
        'n': 'NOUN',
        'v': 'VERB'
    }
    # Iterate synsets.
    for synset in list(wn.all_synsets()):
        # for each synset definition, check for intensity modifying keywords
        for keyword in keywords:
            if keyword in synset.definition():
                intensifiers.extend([
                    (lemma, wnpos_to_universal[synset.pos()])
                    for lemma in synset.lemma_names()
                ])
    # remove more_than_two words intensifiers, and proper nouns. 
    intensifiers = [ (intensifier, universal_tag)
        for (intensifier, universal_tag) in intensifiers
        if intensifier.islower() and intensifier.isalpha()
    ]
    # remove duplicates, and english stopwords
    intensifiers = list(set(intensifiers))
    intensifiers = [ (intensifier, universal_tag)
        for (intensifier, universal_tag) in intensifiers
        if intensifier not in stopwords_en
    ]
    return intensifiers

intensifiers = get_intensifiers()
# Printing intensifiers
print("\nPrinting Example Intensifiers..")
print(random.sample(intensifiers, 10), end="\b, ...]\n")
print("searching for ['pitch', 'dead', 'deathly', 'stark'] ..")
intensifier_words = [ word for word, tag in intensifiers ]
print(" =>", [ e in intensifier_words 
    for e in ['pitch', 'dead', 'deathly', 'stark']
]) # //=> [True, True, True, True]

# ============================================================== #

def print_tags():
    """
    Print words for each universal part-of-speech tag. 
    For designing purposes of this project, shows 5 examples each.
    """
    # For each element, keys are tags, and values are corresponding
    # sets containing words of each tag. 
    tags = defaultdict(set)
    for word, tag in tagged_words:
        tags[tag].add(word)
    # Dictionary for matching abbreviation
    abbr_to_full = {
        '.': 'Punctuation',     # 문장부호
        'ADJ': 'Adjective',     # 형용사
        'ADP': 'Adposition',    # 전치사, 후치사
        'ADV': 'Adverb',        # 부사
        'CONJ': 'Conjunction',  # 접속사
        'DET': 'Determiner',    # 관형사
        'NOUN': 'Noun',         # 명사
        'NUM': 'Number',        # 수사
        'PRON': 'Pronoun',      # 대명사
        'PRT': 'Predeterminer', # 전치 한정사
        'VERB': 'Verb',         # 동사
        'X': 'Unknown',         # 분류 없음
    }
    # Print words.
    print("\n=== %d Universal Tags ===" % len(tags))
    keys = sorted(tags.keys())
    print(": {" + ", ".join(keys) + "}")
    for i, key in enumerate(keys):
        print("%d. %s (%s):" % (i+1, abbr_to_full[key], key), end=" ")
        print(random.sample(tags[key], 5), end="\b, ... ]\n")


# ============================================================== #
# ==================== Process Collocations ==================== #
# ============================================================== #

# === Scheme 1 ================================================= #
def get_collocations():
    """
    Process collocation_list directly from frequency distribution of bigrams
    in NLTK brown corpus. For each collocation bigram, remove those that contain 
    part-of-speech which obviously do not restrict the meaning of other word.
    Returns list of collocation pairs. 
    [
        ((word, pos), (word, pos)), ...
    ]
    """
    # The term 'collocation' refers to frequent occurrence of phrase 
    # of words that go together. It will be restricted to bigrams here.
    frequentBigrams = [ tagged_bigram
        for tagged_bigram in list(freqDist)
        if freqDist[tagged_bigram] >= 5
    ]
    # Exclude stopwords, and all non-alphabeticals (ex. 'CS372').
    collocation_list = [ ((word1, tag1), (word2, tag2))
        for (word1, tag1), (word2, tag2) in frequentBigrams
        if word1.isalpha() and word2.isalpha()
        if not word1 in stopwords_en
        if not word2 in stopwords_en
    ]
    # Add pos filters. 
    collocation_list = [
        ((word1, tag1), (word2, tag2))
        for (word1, tag1), (word2, tag2) in collocation_list

        # Remove adpositions (prepositions + postpositions), and unknown
        # pos tag words, predeterminers, conjunctions, and pronouns. 
        # ex. (ADP) 'looked like', 'door behind', (X) 'et al', 'deja vue', 
        #     (PRT) "'all' the time", "'both' our children"
        #     (CONJ) 'yet even', (PRON) 'reminds us'
        if not (tag1 == 'ADP' or tag2 == 'ADP')
        if not (tag1 == 'PRT' or tag2 == 'PRT')
        if not (tag1 == 'X' or tag2 == 'X')
        if not (tag1 == 'CONJ' or tag2 == 'CONJ')
        if not (tag1 == 'PRON' or tag2 == 'PRON')

        # Above is for any existence of ADP, PRT, X, CONJ, PRON. 
        # Below removes occurences of any combinations of pos tags, that do
        # nothing to restrict or that only restrict the meaning but do not intensify.
        # 
        #  - Possible pos left: ADJ, ADV, DET, NOUN, NUM, VERB. 
        #  - No restrictions of tag1 == 'ADV' will be given. 
        #  - Inspect all 30 combinations..
        # Case: _, 'ADJ' (ex. 'nothing less', 'made possible', 'one last')
        if not (tag1 in ['DET', 'NOUN', 'NUM', 'VERB'] and tag2 == 'ADJ')
        # Case: _, 'ADV'
        if not (tag1 in ['ADJ', 'DET', 'NOUN', 'NUM'] and tag2 == 'ADV')
        # Case: _, 'DET' (ex. 'one another')
        if not (tag1 in ['ADJ', 'DET', 'NOUN', 'NUM', 'VERB'] and tag2 == 'DET')
        # Case: _, 'NOUN' (ex. 'every day', 'dissolved oxygen', 'stock market')
        if not (tag1 in ['DET', 'NOUN', 'VERB'] and tag2 == 'NOUN')
        # Case: _, 'NUM' (ex. 'first one', 'two hundred')
        if not (tag1 == 'NUM' and word1 not in ['thousand', 'million'])
        if not (tag2 == 'NUM')
        # Case: _, 'VERB' (ex. 'girl said', 'would happen', 'one could')
        if not (tag1 in ['ADJ', 'DET', 'NOUN', 'NUM', 'VERB'] and tag2 == 'VERB')
    ]
    return collocation_list


# === Scheme 2 ================================================= #
def get_collocations2(intensifiers):
    """
    Process collocations_list from given intensifiers with bigrams. 
    Returns list of collocation pairs. 
    [
        ((word, pos), (word, pos)), ...
    ]
    """
    # Initialize collocation_list and frequency_barrier
    collocation_list = []
    frequency_barrier = 3
    # Filter out short comparatives. 
    intensifiers = [(intensifier, tag) for (intensifier, tag) in intensifiers if len(intensifier) >= 6]
    for intensifier, tag in intensifiers:
        # Use intensifiers at front or at back. 
        upright_collocating_pairs = [ ((intensifier, tag), tagged_word)
            for tagged_word, freq in list(conditionalFreqDist[(intensifier, tag)].items())
            if freq >= frequency_barrier
        ]
        reversed_collocating_pairs = [ (tagged_word, (intensifier, tag))
            for tagged_word, freq in list(oppositeConditionalFreqDist[(intensifier, tag)].items())
            if freq >= frequency_barrier
        ]
        collocation_list.extend(upright_collocating_pairs)
        collocation_list.extend(reversed_collocating_pairs)
    # Exclude stopwords, and all non-alphabeticals (ex. 'CS372').
    collocation_list = [ ((word1, tag1), (word2, tag2))
        for (word1, tag1), (word2, tag2) in collocation_list
        if word1.isalpha() and word2.isalpha()
        if not word1 in stopwords_en
        if not word2 in stopwords_en
    ]
    collocation_list = list(set(collocation_list))
    return collocation_list

# ============================================================== #


# ============================================================== #
# ========================= Evaluation ========================= #
# ============================================================== #

def add_restrictivity_score(element):
    """
    Calculate the likelihood of occurence of word1, before word2 or 
    occurence of word2, after word1 based on the mode. 
    Returns element with restrictivity score. 
    """
    front, back = element
    # Measure 1: add scores for frequent bigrams
    _max = 9717 # => max([f for e, f in freqDist.items()])
    restrictivity_score = freqDist[element] / _max
    # Measure 2: calculate upright, and reversed restrictivity. 
    def get_restrictivity(word1, word2, mode="upright"):
        """
        Default mode is "upright", and set to "reversed" if it is using
        oppositeConditionalFreqDist. 
        Return the restrictivity score.
        """
        is_upright = mode == "upright"
        cfd = conditionalFreqDist[word1] if is_upright \
                else oppositeConditionalFreqDist[word2]
        total = sum([freq for word, freq in list(cfd.items())])
        if total == 0: return 0
        # multiplied by 100 to prevent rounding error.
        return (cfd[word2] / total) if is_upright \
                else (cfd[word1] / total)
    # Return with restrictivity_score.
    upright_restrictivity = get_restrictivity(front, back)
    reversed_restrictivity = get_restrictivity(front, back, "reversed")
    restrictivity_score *= upright_restrictivity * reversed_restrictivity
    return restrictivity_score, (front, back)


def add_intensity_score(element):
    """
    If the pair in element contains intensity-modifying words, 
    adds higher score for intensity. 
    Returns element with intensity score. 
    """
    restrictivity_score, ((word1, tag1), \
        (word2, tag2)) = element
    # Measure 1: add scores for existence of intensifiers.
    intensity_score = 1
    for word in [word1, word2]:
        if word in intensifiers:
            intensity_score *= 2
    # Measure 2: use synset definitions for additionals.
    def get_definition_intensity(word):
        definitions = [synset.definition() for synset in wn.synsets(word)]
        counts = [1] * len(definitions) # list to store counts for each definitions. 
        for index, definition in enumerate(definitions):
            count = 1 # Initialize
            tokens = nltk.word_tokenize(definition)
            for token in tokens:
                if token in intensifiers: count += 1
            counts[index] = count
        return max(counts) if not len(counts) == 0 else 1
    intensity_score *= get_definition_intensity(word1)
    intensity_score *= get_definition_intensity(word2)
    return restrictivity_score, intensity_score, \
        ((word1, tag1), (word2, tag2))


def evaluate(collocation_list):
    """
    Evaluate each collocation pairs. 
    Find the best restrictive and intensity-modifying pairs with self-evaluated scores.
    We provide two measures: 
        1. 'restrictivity' / 2. 'intensity'
    Return collocation pair, bind with score in tuple.
    [
        (restrictivity_score * intensity_score, (word1, word2)), 
        ...
    ]
    """
    # Restrictivity Score
    result = map(add_restrictivity_score, collocation_list)
    # Intensity Score
    result = map(add_intensity_score, result)
    # Calculate overall score.
    result = [
        (restrictivity_score * intensity_score, (word1, word2))
        for restrictivity_score, intensity_score, \
            ((word1, tag1), (word2, tag2)) in result
    ]
    return sorted(result)[::-1]

# ============================================================== #


def save(result):
    """
    Save output as a csv file.
    """
    file = open('CS372_HW2_output_20170305.csv', 'w')
    for idx, (_, (word1, word2)) in enumerate(result):
        index = "(%d)" % (idx + 1)
        file.write(",".join([index, word1, word2]) + "\n")
    file.close()


# Main function for word processing algorithm. 
def main():
    """
    For this assignment, collocation lists will be processed in 
    two different ways of schemes.
    """
    # print universal tags
    print_tags()

    # Process collocation list
    print("\nRunning 'intensity', 'restrictivity' schemes..")
    collocation_list1 = get_collocations()
    collocation_list2 = get_collocations2(intensifiers)

    # evaluate processed collocation list 
    print("Evaluating collocation_lists..")
    result1 = evaluate(collocation_list1)[:50]
    result2 = [e for e in evaluate(collocation_list2) if e not in result1][:50]
    
    print("\n=== Results ===")
    result = result1 + result2
    pprint.pprint(result)

    # save as a csv file
    save(result)


if __name__ == "__main__":
    main()
