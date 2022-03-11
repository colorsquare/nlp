#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown, stopwords
import random, pprint

# === Global Variables === #
# brown text corpora
text = brown.words()
stopwords_en = stopwords.words('english')
bigrams = nltk.bigrams(text)

# nltk frequency distribution
freqDist = nltk.FreqDist(text)
conditionalFreqDist = nltk.ConditionalFreqDist(bigrams)
oppositeConditionalFreqDist = nltk.ConditionalFreqDist(
    (back, front) for front, back in bigrams
)
# ======================== #

def get_adverbs():
    """
    Process intensity-modifying adverbs, from NLTK wordnet corpus. 
    Returns list of adverbs. 
    """
    # keywords from definition of 'highly' and 'very'
    keywords = ['extent', 'intensifier', 'intensity', 'quantifier', 'degree', 'comparative']  # decides which adverbs to get. 
    adverbs = []
    for synset in list(wn.all_synsets('r')):
        # for each synset definition, check for intensity modifying keywords
        hold_count = 0  # number of keywords that this definition is holding. 
        for keyword in keywords:
            if keyword in synset.definition():
                hold_count += 1
        # add lemmas in synset only if definition holds more than two keywords.
        if hold_count >= 2: adverbs.extend(synset.lemma_names())
    
    # remove more_than_two words adverbs
    adverbs = [adverb for adverb in adverbs if "_" not in adverb]
    # remove duplicates, and english stopwords
    adverbs = list(set(adverbs) - set(stopwords_en))
    
    # add additional filter to get most frequent adverbs, and remove short comparatives
    filtered_adverbs = [adverb for adverb in adverbs if freqDist[adverb] >= 10 and len(adverb) >= 7]
    return filtered_adverbs


def get_pairs(adverbs):
    """
    From the given adverbs, get words that are utilized most often. 
    Returns list of pairs element. 
    [
        ('v', adverb, ( verb, count ) ), 
        ('a', adverb, ( adjective, count ) ), 
        ...
    ]
    """
    pairs = []
    used_words = [] # for diversity in result
    for adverb in adverbs:
        # collocation words of adverb
        collocations = list(conditionalFreqDist[adverb].items())
        # filter collocations with following criteria
        # 1. exclude punctuations / 2. exclude stopwords / ( 3. count ) can be used or not. 
        collocations = [(word, count) 
            for word, count in collocations
            if word.isalpha()
            if word not in set(stopwords_en)
        ]
        # only extract verbs, adjectives
        for word, count in collocations:
            if word in used_words: continue
            part_of_speeches = [synset.pos() for synset in wn.synsets(word)]
            if 'v' in part_of_speeches: 
                pairs.append(('v', adverb, (word, count)))
            if 'a' in part_of_speeches: 
                pairs.append(('a', adverb, (word, count)))
            used_words.append(word)
    return pairs


def get_synsets(pairs):
    """
    From the adverb, collocation_word pairs, find and process synsets. 
    Return pairs element with synonyms attached. 
    """
    synseted = []
    # iterate verbs, adjectives and find their synonyms
    for pos, adverb, (word, count) in pairs:
        synsets = wn.synsets(word)
        synonym_list = [synset.lemma_names()
            for synset in synsets 
            if synset.pos() is pos 
            if len(synset.lemma_names()) >= 4
        ]
        synonym = synonym_list[0] if len(synonym_list) != 0 else []
        if len(synonym) != 0:
            elem = (pos, adverb, (word, count), synonym)
            synseted.append(elem)
    return synseted


def evaluate(adverbs, pairs):
    """
    Evaluate each pairs element. 
    Find the best matching synonyms with self-evaluated scores. 
    Return pairs to save, with score.
    [
        (score, 'v', adverb + verb, best_scored_word), 
        (score, 'a', adverb + adjective, best_scored_word), 
        ...
    ]
    """
    evaluated = []
    for pos, adverb, (word, _), synonym in pairs:
        # list to contain scores of each synonym
        scored_synonym = [(0, syn) for syn in synonym]
        for index, (score, syn) in enumerate(scored_synonym):
            # lower scores for similar words (to exclude difference in tense)
            word_letters = list(word)
            for letter in syn:
                if letter not in "aeiou" and letter in word_letters:
                    word_letters.remove(letter)
            common_length = len(word) - len(word_letters)
            score = 1 - (common_length / len(syn))
            # lower scores for words that collocate with adverb a lot.
            collocation_freq = 0
            for collocation, _ in oppositeConditionalFreqDist[syn].items():
                if collocation in adverbs: collocation_freq += 1
            if collocation_freq != 0:
                score *= 1 / (1 + collocation_freq)
            # give extra score to diversity of synonym, 
            # because if is likely to have varied meanings. 
            score *= len(synonym) / 28
            # save score
            scored_synonym[index] = (score, syn)
        # get the best score and scored word
        score, best_scored_word = sorted(scored_synonym)[::-1][0]
        elem = (score, pos, adverb + " " + word, best_scored_word)
        evaluated.append(elem)
    
    # sort the evaluated pairs
    evaluated = sorted(evaluated)[::-1]
    return evaluated


def save(pairs):
    """
    Save output as a csv file and print out statistics. 
    Return none.
    """
    num_verbs = 0 # how many resulting verbs
    # count for how may pairs to save
    count = 50
    file = open('CS372_HW1_output_20170305.csv', 'w')
    for index, (_, pos, collocation, word) in enumerate(pairs):
        if pos == 'v': num_verbs += 1
        if index >= (count-1): continue
        file.write(",".join([collocation, word]) + "\n")
    file.close()
    # statistics
    print("Total %d pairs extracted." % len(pairs))
    print("    Number of verb pairs: %d" % num_verbs)
    print("    Number of adjective pairs: %d" % (len(pairs) - num_verbs))
    print("=> %5.2f%% are verbs" % (num_verbs / len(pairs)))


# Main function of word processing algorithm. 
def main():
    # extract adverbs list with seed 'highly' and 'very'
    adverbs = get_adverbs()
    # print
    print("(ADVERBS) Total %d extracted: " % len(adverbs))
    print(', '.join(sorted(adverbs)))

    # get adverb, and word pairs, and synsets
    pairs = get_pairs(adverbs)
    synseted_pairs = get_synsets(pairs)
    # print
    print("(SYNONYMS) Total %d pairs: " % len(synseted_pairs))
    pprint.pprint(synseted_pairs[:20])
    print("\b, ...]")

    # evaluate processed word pairs
    evaluated_pairs = evaluate(adverbs, synseted_pairs)
    # print
    random.shuffle(evaluated_pairs)
    print("(EVALUATED) Total %d pairs: " % len(evaluated_pairs))
    print("[")
    for index, (score, pos, collocation, word) in enumerate(evaluated_pairs):
        print("  %s, %s (%5.3f) '%s'," % (collocation, word, score, pos.upper()))
        if index >= 20: 
            print('  ...\n]')
            break

    # save as a csv file
    sorted(evaluated_pairs)
    save(evaluated_pairs)


if __name__ == "__main__":
    main()
