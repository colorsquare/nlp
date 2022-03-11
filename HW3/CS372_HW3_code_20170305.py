#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NLTK
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown, cmudict, stopwords
from nltk.stem import WordNetLemmatizer
# Network
from urllib import request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
# Misc
import time, os
from collections import defaultdict
from pprint import pprint


# Globals
heteronym_keys = []
heteronyms = dict()
stopwords_en = stopwords.words('english')
cdict = cmudict.dict()
tagged_sentences = brown.tagged_sents(tagset="universal")
n = WordNetLemmatizer()


def get_heteronyms():
    """Get heteronym list.

    If `heteronyms.txt` exists, parse it and return. 
    Else, build heteronym list.

    Updates:
        heteronyms (Dictionary)
        heteronym_keys (List)
    """
    file_name = 'heteronyms.txt'
    if not os.path.isfile(file_name):
        # Candidate for heteronyms
        candidates = [ word
            for word, pronounciations in cdict.items()
            # Has different pronounciations
            if len(pronounciations) >= 2
            # Has more than one meaning
            if len(wn.synsets(word)) >= 2
        ]

        # Web crawling
        file = open(file_name, "a")
        for idx, word in enumerate(candidates):
            entry = get_heteronym_entry(word)
            # Insert to dictionary
            if entry:
                file.write("".join([word, ": ", str(entry), "\n"]))
                heteronyms[word] = entry
            time.sleep(1) # HTTPError code 429: Too Many Requests
        file.close()
    else:
        heteronym_file = open(file_name, 'r')
        lines = [
            line.strip()
            for line in heteronym_file.readlines()
        ]
        for line in lines:
            split_idx = line.find(': ')
            word = line[:split_idx]
            entry = eval(line[split_idx + 2:])
            heteronyms[word] = entry
        heteronym_file.close()

    # exclude heteronyms only with abbreviations
    remove = []
    for heteronym in heteronyms:
        entries = heteronyms[heteronym]
        for idx, entry in enumerate(entries):
            pron, list_ = entry
            new_list_ = [ e
                for e in list_
                if e[0] != 'abbreviation'
            ]
            entries[idx] = (pron, new_list_)
        new_entries = [ entry
            for entry in entries
            if len(entry[1]) != 0
        ]
        heteronyms[heteronym] = new_entries
        if len(new_entries) == 1 or heteronym in stopwords_en:
            remove.append(heteronym)
    for e in remove:
        del heteronyms[e]
    
    heteronym_keys.extend(list(heteronyms.keys()))


def get_heteronym_entry(word):
    """Web crawler to verify heteronym.

    Args:
        word (String): candidate `word` for heteronym. 
    
    Returns:
        results (List): Build heteronym entry. List of pronounciations dictionary, 
            with each entry as list of part-of-speech and meaning tuple. 
            If no entry exists, return empty list. 
            [
                (
                    pronounciation: [
                        (part-of-speech1, meaning1-1),
                        (part-of-speech1, meaning1-2),
                        (part-of-speech2, meaning2-1),
                        ...
                    ], ...
                ),
                ...
            ]

    Raises:
        Exception: HTTPError except 404. 
        AttributeError: No required attribute for word in web.
    """
    # urllib
    url = "https://www.lexico.com/en/definition/"
    try:
        page = request.urlopen(url + word)
    except HTTPError as e:
        if e.code == 404:
            return []
        raise Exception
    html = page.read().decode('utf8')

    # beautifulsoup
    soup = BeautifulSoup(html, 'html.parser')
    homographs = soup.find_all(class_="entryGroup")

    # not a heteronym
    if len(homographs) <= 1: return []

    # html content elements
    contents = soup.find(class_="entryWrapper").contents
    locations = [
        contents.index(homograph)
        for homograph in homographs
    ]

    # find pronounciation, part-of-speech, and meaning.
    result = dict()

    # exclude ones with all same pronounciation
    try:
        pronounciations = [
            homograph.find_all(class_="phoneticspelling")[0].get_text()
            for homograph in homographs
        ]
        # not a heteronym, but homograph
        if len(set(pronounciations)) == 1: 
            return []

        for idx, homograph in enumerate(homographs):
            # grambs: for each pronounciation, for each part-of-speech
            start_location = locations[idx]
            end_location = locations[idx+1] if idx < len(homographs) - 1 else len(contents)
            grambs = [
                content
                for content in contents[start_location:end_location]
                if 'class="gramb"' in str(content)
            ]
            
            # add each part-of-speech with meaning.
            meaning_list = []
            for gramb in grambs:
                pos = gramb.h3.find(class_="pos").get_text()
                
                # search only direct children
                meaning_list.extend([ 
                    (
                        pos,
                        meaning.find(class_="trg").p.find(class_="ind").get_text()
                    )
                    for meaning in gramb.ul.find_all("li", recursive=False)
                ])

            # insert to existing pronounciation, else create new entry.
            try:
                existing_list = result[pronounciations[idx]]
                result[pronounciations[idx]] = existing_list + meaning_list
            except KeyError:
                result[pronounciations[idx]] = meaning_list
    except AttributeError:
        return []

    # close and return entry
    page.close()
    return list(result.items())


def search_heteronyms():
    """Find and evaluate heteronyms for all sentences. 

    Returns:
        answer (List): Sorted in order of high score.
        [
            (score, result), ...
        ]
    """
    answer = []
    for sentence in tagged_sentences:
        score, result = evaluate(sentence)
        if score != 0:
            result.insert(1, "Brown")
            answer.append((score, result))
    return sorted(answer)[::-1]


def evaluate(sentence):
    """Evaluate each sentence. 

    Several schemes are applied. 
        1. More occurrence of heteronyms gives higher score. 
        2. More occurrence of heteronyms with same letters gives higher score. But, 
            it is only for different pronounciations. Lower score is given for same
            pronounciations. 
        3. More occurrence of heteronyms with same letters and same part-of-speech
            gives higher score. 
    
    Returns:
        score (Integer): score according to above schemes. 
        result (List): [
            sentence, 
            "word1/pronounciation1/pos1",
            "word2/pronounciation2/pos2",
            ...
        ]
    """
    count = 0
    occurrence = defaultdict(list)
    result = [
        " ".join([word for word, pos in sentence])
    ]
    for idx, (word, pos) in enumerate(sentence):
        # not heteronym
        lemmatized_word = n.lemmatize(word.lower(), 'v') if pos == "VERB" else word.lower()
        if not lemmatized_word in heteronym_keys:
            continue
        # scheme 1: increment count
        count += 1
        # find matching pronounciation, pos, definition in heteronym entry
        het_pro, het_pos, _ = find_matching_heteronym(idx, sentence)
        result.append(word + het_pro + het_pos)
        # for scheme 2, 3
        occurrence[word.lower()].append((het_pro, het_pos))
    
    # scheme 2: additional score for heteronyms with same letters
    homographs = []
    for word, items in list(occurrence.items()):
        fd = nltk.FreqDist([ het_pro
            for het_pro, _ in items
        ])
        homographs.append(len(het_pro) / fd.most_common(1)[0][1])

    # scheme 3: additional score for heteronyms with same letters and same part-of-speech
    homoposes = [
        len(het_poses) - len(set(het_poses))
        for word, het_poses in list(occurrence.items())
        if len(het_poses) - len(set(het_poses))
    ]

    def calculate(list_):
        value = 1
        for num in list_:
            value *= 2 ** num
        return value

    # calculate score, make result
    score = (4 ** count) * calculate(homographs) * calculate(homoposes) / len(sentence) if count else 0
    return score, result


def find_matching_heteronym(idx, sentence):
    """Find matching heteronym for the word in sentence. 

    Consult pos given along with word in sentence, and 
    context and meaning is compared to find right `match`.

    Args:
        idx (Integer): idx for the word we want to match. 
        sentence (List): sentence to get context and pos. 
    
    Returns: 
        het_pro (String): heteronym pronounciation. 
        het_pos (String): heteronym part-of-speech. 
        het_def (String): heteronym definition. 
    """
    # all heteronym entry
    word, pos = sentence[idx]
    lemmatized_word = n.lemmatize(word.lower(), 'v') if pos == "VERB" else word.lower()
    entries = heteronyms[lemmatized_word]

    # heteronym pos to universal
    het_pos_to_universal = {
        '': 'X', 
        'ordinal number': 'NUM', 
        'adjective': 'ADJ', 
        'intransitive verb': 'VERB', 
        'adverb': 'ADV', 
        'verb': 'VERB', 
        'noun': 'NOUN', 
        'transitive verb': 'VERB',
        'auxiliary verb': 'VERB',
        'conjunction': 'CONJ',
        'preposition': 'ADP',
        'pronoun': 'PRON',
        'proper noun': 'PROPERNOUN', 
        'abbreviation': 'ABBR'
    }
    
    # choose matching pronounciation, pos, definition..
    if idx != 0 and not word.islower():
        find = []
        for het_pro, _list in entries:
            for het_pos, het_def in _list:
                if het_pos == 'proper noun':
                    find.append((het_pro, het_pos, het_def))
        if find:
            return find[0]
    else:
        find = []
        for het_pro, _list in entries:
            for het_pos, het_def in _list:
                if het_pos_to_universal[het_pos] == pos:
                    find.append((het_pro, het_pos, het_def))
        if find:
            for idx, element in enumerate(find):
                het_pro, het_pos, het_def = element
                count = 0
                for item in het_def:
                    if item in sentence: count += 1
                find[idx] = (2 ** (len(find) - idx) + count, element)
            return sorted(find)[::-1][0][1]

    # default match to first one.
    het_pro, _list = entries[0]
    het_pos, het_def = _list[0]
    return het_pro, het_pos, het_def


def save(answer):
    file = open('CS372_HW3_output_20170305.csv', 'w')
    for score, result in answer:
        file.write(",".join(result) + "\n")
    file.close()


# Main function for word processing algorithm. 
def main():
    # Build heteronym list
    get_heteronyms()

    # Find sentences with heteronyms
    answer = search_heteronyms()
    pprint(answer[:30])

    # Save to csv
    save(answer[:30])


if __name__ == "__main__":
    main()
