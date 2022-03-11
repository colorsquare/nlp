import nltk
import re, random
from pprint import pprint


def get_tagged_sentences():
    """Reads pre-processed tagged sentences.

    Returns:
        sentences (list): each item follows below format;
            (
                sentence_text, 
                [(X, Action, Y), ..]
            )
    """
    f = open('CS372_HW4_output_20170305.csv', 'r')
    lines = f.readlines()

    # parse sentences
    train = []
    test = []
    text = None
    dataType = "train"
    for line in lines:
        if line.startswith("Text"):
            text = line.strip()[6:]
        elif line.startswith("DataType"):
            dataType = "test" if "test" == line.strip()[10:] else "train"
        elif line.startswith("Tags"):
            if dataType == "train":
                train.append((text, eval(line.strip()[6:])))
            else:
                test.append((text, eval(line.strip()[6:])))
        else: continue
    f.close()
    return train, test


def additional_tags(elem):
    """Modify tags for certain words."""
    word, tag = elem
    if word.lower() == 'but': tag = "BUT"
    if word.lower() == 'that': tag = "THAT"
    if word.lower() == 'whether': tag = "WHETHER"
    if word.lower() == 'although': tag = "ALTHOUGH"
    if word.lower() == 'while': tag = "WHILE"
    if word.lower() == 'for': tag = "FOR"
    return (word, tag)


def chunk(sentence_text):
    """Chunk the given sentence text.

    Args:
        sentence_text (string): given sentence 
            in plain text.

    Returns:
        chunked_sentence (tree): chunked by 
            predefined syntax with RegexpParser.
    """
    syntax = r"""
        # Conjuctions
        CONJ: {<RB><,>}
        # Noun Phrase
        NP: {<DT|PRP\$>? (<JJ.*><CC>)* <CD|JJ.*|VBG|VBN|RB.?>* <NN.*|VBG|CD|POS|PRP>+  (<\(>(<CC>?<NN.*|JJ>+)+<\)>)?}
            {<PRP|EX>}
        # Verb Phrase
        VP: {<VB|VBP|VBZ|VBD><RB.?>?(<VBN><RB.?>?<IN|TO>)?}
            }<VBG>{  # chinking
        # Multiple Noun Phrase
        MNP: {<NP> (<,><NP>)+ (<,><CC><NP>)}
             {<NP> (<CC><NP>)?}
        MNP: {<MNP><MNP>+}
        # Preposition Phrase
        INP: {<IN><RB.?>?<MNP>}
        TOP: {<TO><RB.?>?<MNP|VP>}
        # Clause
        CLAUSE: {<MNP|W.*> <RB.?>? <INP|TOP>* <RB.?>? <VP> <RB.?>? <MNP>* <RB.?>? <INP|TOP>* <RB.?>? (<CC|,>? <RB.?>? <VP><MNP>* <RB.?>? <INP|TOP>* <RB.?>?)*}
        # How to distinguish preposition and subordinating conjunction?
        # That Phrase
        THATP: {<THAT><CLAUSE>}
        # Whether Phrase
        WHETHERP: {<WHETHER><CLAUSE>}
    """
    # Chunker
    parse_chunker = nltk.RegexpParser(syntax, loop=2)

    # Tokenize, pos_tag, and chunk.
    tokens = nltk.word_tokenize(sentence_text)
    tagged = list(map(additional_tags, [
        (word, tag)
        for word, tag in nltk.pos_tag(tokens)
        # if not re.match(r"RB.*", tag)
        if not re.match(r"MD", tag)  # remove modals
    ]))
    return parse_chunker.parse(tagged)


def extract(chunked_sentence):
    """Extract relations <X, Action, Y> from chunked tree. 

    Args:
        chunked_sentence (tree): chunked sentence 
            expressed in tree.

    Returns:
        relations (list): [(X, Action, Y), ..]
    """
    def traverse(t):
        relations = []
        X = ""
        Action = ""
        for idx, elem in enumerate(t):
            try:
                elem.label()
            except AttributeError:
                # save which as X
                if elem[0] in ['which']:
                    X = elem[0]
            else:
                if elem.label() == 'MNP':
                    MNP = get_text(elem)
                    if Action:
                        relations.append((X, Action, MNP))
                    else:
                        if not X:
                            X = MNP
                    Action = ""
                elif elem.label() == 'VP':
                    verb = get_text(elem, True)
                    if is_action(verb):
                        Action = verb
                elif elem.label() in ["CLAUSE", "THATP", "WHETHERP", "S"]:
                    child_relations = traverse(elem)
                    # search for which and its subject
                    for child_idx, child_relation in enumerate(child_relations):
                        child_x, child_action, child_y = child_relation
                        if child_x == 'which':
                            child_relations[child_idx] = \
                                (get_closest_mnp(t, idx), child_action, child_y)
                    relations.extend(child_relations)
        return relations

    def get_closest_mnp(t, idx):
        idx -= 1
        while idx >= 0:
            closest = t[idx]
            try:
                closest.label()
            except AttributeError:
                idx -= 1
            else:
                if closest.label() == 'MNP':
                    return get_text(closest)
                else:
                    MNP = get_closest_mnp(closest, len(closest))
                    if MNP != 'which':
                        return MNP
                    else:
                        idx -= 1
        # MNP not found
        return 'which'

    def is_action(verb):
        action_list = ['activate', 'inhibit', 'bind', 'stimulate', 'prevent']
        for action in action_list:
            if action in verb:
                return True
        return False

    def get_text(t, is_vp = False):
        words = []
        for elem in t:
            try:
                elem.label()
                words.append(get_text(elem, is_vp))
            except AttributeError:
                if is_vp:
                    if not re.match(r"RB.*", elem[1]):
                        words.append(elem[0])
                else:
                    words.append(elem[0])
        return join(words)

    def join(words):
        answer = ""
        prev_was_open_braket = False
        for idx, word in enumerate(words):
            if word in [')', ',', '.', "'"]:
                answer += word
            elif word in ['(']:
                prev_was_open_braket = True
                answer += " " + word
            else:
                if idx == 0:
                    answer += word
                elif prev_was_open_braket:
                    answer += word
                    prev_was_open_braket = False
                else:
                    answer += " " + word
        return answer

    relations = traverse(chunked_sentence)
    return relations


def evaluate(result):
    """Evaluate extracted relations.

    Args:
        result (list): list of tuples;
            [
                (
                    extracted_relations,
                    answer_relations
                ),
                ...
            ]
            extracted_relations, answer_relations;
            [(X, Action, Y), ..]

    Prints:
        Precision (float): TP / (TP + FP)
        Recall (float): TP / (TP + FN)
        F-score (float): (2 * Precision * Recall) / (Precision + Recall)
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for idx, relation_pair in enumerate(result):
        extracted, answer = relation_pair
        if not extracted:
            false_negative += 1
        elif extracted != answer:
            false_positive += 1
        else:
            true_positive += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = (2 * precision * recall) / (precision + recall)
    print("Precision: %6.2f" % precision)
    print("Recall: %9.2f" % recall)
    print("F-Score: %8.2f" % f_score)


def main():
    # divide sentences into train, and test sets.
    train, test = get_tagged_sentences()

    # build chunker and relation extraction module
    # from train data with manual modification.
    train_result = []
    for sentence in train:
        text, triples = sentence
        chunked_sentence = chunk(text)
        relations = extract(chunked_sentence)
        train_result.append((relations, triples))

    # evaluate training
    print("[Training]")
    evaluate(train_result)

    # input test data
    result = []
    for sentence in test:
        text, triples = sentence
        chunked_sentence = chunk(text)
        relations = extract(chunked_sentence)
        result.append((relations, triples))

    # evaluate
    print("[Testing]")
    evaluate(result)


if __name__ == "__main__":
    main()
