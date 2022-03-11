import nltk, re, wikipediaapi
from nltk.tree import Tree
from nltk.corpus import names
from collections import deque
from pprint import pprint


# wikipedia API
wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)


def parse_gap():
    """Parse GAP reference datasets

    Returns:
        development, test, validation (List): each lists all contain
            same element form as below. 
            [
                (
                    text, pronoun, pronoun-offset, 
                    A, A-offset, A-coref, 
                    B, B-offset, B-coref, 
                    URL
                )
                , ...
            ]
    """
    development = []
    f = open("gap-coreference/gap-development.tsv", 'r')
    f.readline()  # Remove first line
    for line in f.readlines():
        elem = tuple(line.strip().split('\t')[1:])
        development.append(elem)
    f.close()

    test = []
    f = open("gap-coreference/gap-test.tsv", 'r')
    f.readline()  # Remove first line
    for line in f.readlines():
        elem = tuple(line.strip().split('\t')[1:])
        test.append(elem)
    f.close()

    validation = []
    f = open("gap-coreference/gap-validation.tsv", 'r')
    f.readline()  # Remove first line
    for line in f.readlines():
        elem = tuple(line.strip().split('\t')[1:])
        validation.append(elem)
    f.close()

    return development, test, validation


def filter(sentence):
    """Filter given sentence before any processing.

    Args:
        sentence (List): each sentence from 'tokenize'.
    
    Returns:
        filtered (List): removes elements according to 
            filter below.
    """
    filtered = [(word, tag)
        for word, tag in sentence
        # if not re.match(r"RB.*", tag)  # remove adverbs
        # if not re.match(r"MD", tag)  # remove modals
    ]
    return filtered


def tokenize(text):
    """Returns text tokenized as sentences with nltk.

    Split into sentences, then words, and annotate with part-of-speech 
    tags. Next, each sentence is filtered by 'filter'.
    """
    return [
        filter(nltk.pos_tag(nltk.word_tokenize(sent)))
        for sent in nltk.sent_tokenize(text)
    ]


def annotate_snippet(item):
    """Annotate snippet of pronoun, A, B.

    Returns:
        sentences (List of List): result by function 'tokenize'
        indexes (List of Tuples): Locates the indexes of desired pronoun, 
            name A, name B of sentences above. Each element is shown as 
            tuple, with sent_index, word_index, and length. 
            [
                (sent_index, word_index, length),  # pronoun
                ... # name A, name B
            ]
        answer (Tuple of Booleans): whether name A, name B is a reference of the given 
            pronoun. (coreference of name A, coreference of name B)
        url (String): wikipedia url used for getting page context.
    """
    text, pron, pron_offset, name_a, a_offset, a_coref, name_b, b_offset, b_coref, url = item

    # initialize return values
    sentences = tokenize(text)
    indexes = [None, None, None]
    answer = (a_coref == "TRUE", b_coref == "TRUE")

    # offsets to simple indexes
    simple_indexes = [
        sum([len(sent) for sent in tokenize(text[:int(pron_offset)])]),
        sum([len(sent) for sent in tokenize(text[:int(a_offset)])]),
        sum([len(sent) for sent in tokenize(text[:int(b_offset)])]),
    ]

    # find name A, name B. 
    index_count = 0
    for sent_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):

            # check if pronoun, name A, or name B.
            for idx, simple_index in enumerate(simple_indexes):
                if simple_index == index_count:
                    word_length = 0
                    if idx == 0:
                        word_length = sum([len(sent) for sent in tokenize(pron)])
                    elif idx == 1:
                        word_length = sum([len(sent) for sent in tokenize(name_a)])
                    elif idx == 2:
                        word_length = sum([len(sent) for sent in tokenize(name_b)])
                    
                    # assert word length exists
                    assert word_length
                    indexes[idx] = (sent_index, word_index, word_length)

            # update simple index
            index_count += 1

    # assert indexes are found
    assert indexes[0] and indexes[1] and indexes[2]
    return sentences, indexes, answer, url


def get_page_context(url):
    """Get page context from the url.

    Args:
        url (String): url in following format
            "http://en.wikipedia.org/wiki/~~"

    Returns:
        text (String): text of wikipedia page. 
            None if page does not exist.
    """
    base_len = len("http://en.wikipedia.org/wiki/")
    page_name = url[base_len:]
    wiki_page = wiki.page(page_name)
    if not wiki_page.exists():
        return None
    return wiki_page.text


def update_annotation(page_text, original_text, indexes):
    """Enlarge text with page_text, update indexes.

    Args:
        page_text (String): page text from wikipedia API
        original_text (String): original snippet text
        indexes (List of Tuples): result by function 'annotate_snippet'

    Returns:
        page_sentences (List of Tuples): result by function 'tokenize'
        updated_indexes (List of Tuples): push indexes according to added page_text
    """
    # find original_text in page_text
    paragraph = None
    found_idx = -1
    for line in page_text.split("\n"):
        idx = line.find(original_text)
        if idx != -1:
            paragraph = line
            found_idx = idx
            break
    assert paragraph

    # update tokenized text, and indexes
    page_tokenized_text = tokenize(paragraph)
    shift = len(tokenize(paragraph[:found_idx]))
    updated_indexes = [
        (sent_index + shift, word_index, length)
        for sent_index, word_index, length in indexes
    ]
    return page_tokenized_text, updated_indexes


def chunk(sentences, indexes):
    """Chunk each sentence using RegexpParser.

    Args:
        sentences (List of List): result by function 'tokenize'
        indexes (List of Tuples): result by function 'annotate_snippet'

    Returns:
        chunked_sentences (Tree): chunked by predefined syntax according 
            to part-of-speech tags.
        chunked_indexes (List of Tuples): updated indexes due to chunking.
            The format changes to ..
            [
                (sent_index, tree_index, length),  # pronoun
                ...  # name A, name B
            ]
            tree_index (Tuple): locate start of the word.
    """
    syntax = r"""
        # Noun Phrase
        NP: {<NNP><CD><,><CD>}  # Date
            {<DT|PRP\$>? (<JJ.?><CC>)*<CD|JJ.?|RB.?>* <NN.?|CD|PRP>+}
        # Multiple Noun Phrase
        MNP: {<NP>*<``><NP><''><NP>*}  # Quotation Mark
             {<NP><\(><NP><\)>}
             {<NP> (<,><NP>)+ (<,><CC><NP>)}
             {<NP> <CC><NP>}
             {<NP><POS>}
        MNP: {<MNP><NP>}  # for joining those with POS
             {<NP>}
        # Preposition Phrase
        PP: {<IN> <RB.?>* <MNP|VP> <RB.?>*}
        # To Phrase
        TP: {<TO> <RB.?>* <MNP|VP> <RB.?>*}
        # Verb Phrase
        V: {<VBD> <RB.?|MD>* <VBN> <RB.?|MD>* <RP>}
           {<MD>?<VB.?>}
        VP: {<V> <RB.?>* <MNP|PP|TP>+}
        # Sentence
        # S {<MNP> <PP|TP>* <VP>}
    """
    # chunker
    parse_chunker = nltk.RegexpParser(syntax, loop=2)

    # separate into sentences
    chunked_sentences = []
    for sentence in sentences:
        chunked_sentence = parse_chunker.parse(sentence)
        chunked_sentences.append(chunked_sentence)

    # locate indexes..
    def find_index(tree, count):
        """Find index of word count.

        Args:
            tree (Tree): tree-like structure by RegexpParser.
            count (Integer): smaller than number of leaves in tree.

        Returns:
            tree_index (Tuple): index of specific leaf in tree.
                ex. (0, 1) - first branch's second leaf.
        """
        curr = 0
        for idx, element in enumerate(tree):
            try:
                element.label()
            except AttributeError:
                if curr == count:
                    return (idx, )
                curr += 1
            else:
                elem_count = word_count(element)
                if count <= curr + (elem_count - 1):
                    recursive_index = find_index(element, count - curr)
                    return (idx, *recursive_index)
                curr += elem_count
        assert False  # not reached

    chunked_indexes = []
    for sent_index, word_index, length in indexes:
        tree_index = find_index(chunked_sentences[sent_index], word_index)
        chunked_indexes.append((sent_index, tree_index, length))
    return chunked_sentences, chunked_indexes


def word_count(tree):
    """Count number of words in tree.
    
    Returns:
        count (Integer): number of words.
    """
    count = 0
    for element in tree:
        try:
            element.label()
        except AttributeError: # leaf
            count += 1
        else: # branch
            count += word_count(element)
    return count


def traverse(tree, index, length = 1):
    """Traverse tree for index. 
    
    This function do not checks for faulty input. Only valid 
    inputs are required. 
    Returns node of length at index.
    """
    for idx, num in enumerate(index):
        if idx == len(index) - 1:
            return tree[num:num+length]
        else:
            tree = tree[num]


def find_s_mnp(sentences, index):
    """Find nearest MNP, or S.

    Args:
        sentences (List of Tree): list of sentence tree 'S'.
        index (List): list of numbers, which is index of sentences.

    Returns:
        path (List): list of visited index.
        index (List): equal to description above.
    """
    path = [index]
    skipped_first = False
    while True:
        index = index[:-1]
        path.append(index)
        # break on encountering s or mnp. 
        element = sentences[:]
        for i in index:
            element = element[i]
        if element.label() == "S": break
        if element.label() == "MNP":
            if skipped_first: break
            skipped_first = True
    return path, index


def breadth_first_search(sentences, path, limit = None, get_first = False, right = False):
    """Breadth-first search of sentences.

    Args:
        sentences (List of Tree): list of sentence Tree 'S'.
        path (List): list of index to start with.
        limit (List): righthand limit to not cross.
    
    Returns:
        candidate (Tree): second-mnp from the search.
            If not found, return as None.
    """
    def is_part_of(tuple_a, tuple_b):
        """Returns if tuple_a is part of tuple_b."""
        if len(tuple_a) > len(tuple_b): return False
        for idx, elem in enumerate(tuple_a):
            if not elem == tuple_b[idx]:
                return False
        return True

    encountered_mnp = True if get_first else False
    curr = tuple()
    root = traverse(sentences, path)[0]
    if right: root = root[::-1]
    # search
    queue = deque([])
    queue.append((root, curr))
    while queue:
        node, curr = queue.pop()
        if right: node = node[::-1]
        for idx, elem in enumerate(node):
            try:
                if limit and (*curr, idx) >= limit: 
                    continue
                if elem.label() == "MNP" and (not limit or is_part_of((*curr, idx), limit)):
                    if encountered_mnp:
                        return elem
                    else:
                        encountered_mnp = True
                queue.append((elem, (*curr, idx)))
            except AttributeError:
                pass
    return None


def valid(candidate, pronoun):
    """Checks if candidate is valid. 

    A valid candidate should be not NULL,
    should not contain pronoun, match right sex.

    Args:
        candidate (Tree): extracted candidate from function
            'breadth_first_search'
        pronoun (String): the pronoun we want to check for.

    Returns:
        is_valid (Boolean)
    """
    def has_pronoun(tree):
        """Returns true if tree contains pronoun."""
        for elem in tree:
            try:
                elem.label()
                if has_pronoun(elem): 
                    return True
            except AttributeError:
                if re.match(r"PRP.?", elem[1]):
                    return True
        return False
    
    def match_sex(candidate, pronoun):
        """Returns true if match sex."""

        def tree_to_list(tree):
            """Returns tree modified to list."""
            answer = []
            for elem in tree:
                try:
                    elem.label()
                    answer.extend(tree_to_list(elem))
                except AttributeError:
                    answer.append(elem[0])
            return answer

        def get_sex(tree):
            """Return sex if exists. If nothing, return None."""
            m = [name.lower() for name in names.words('male.txt')]
            f = [name.lower() for name in names.words('female.txt')]
            for e in tree_to_list(tree):
                if e in m: return "m"
                if e in f: return "f"
            return None

        pronoun_sex = {
            'her': "f",
            'hers': "f", 
            'he': "m", 
            'him': "m", 
            'she': "f", 
            'his': "m"
        }
        # print(pronoun, candidate)
        candidate_sex = get_sex(candidate)
        pronoun = pronoun.lower()
        if not candidate_sex or pronoun_sex.get(pronoun, "-") == candidate_sex:
            return True
        return False

    if not candidate or has_pronoun(candidate) or \
       not match_sex(candidate, pronoun):
        return False
    return True


def hobbs_algorithm(chunked_sentences, sent_tree_index):
    """Search by Hobb's algorithm.

    Args:
        chunked_sentences (Tree): result by function 'chunk', 
            input from function 'extract'.
        sent_tree_index (Tuple): sent_index, tree_index combined.

    Returns:
        coreference (Tree): MNP phrase found as coreference.
    """
    # find pronoun
    node = traverse(chunked_sentences[:], sent_tree_index)
    pronoun = node[0][0]

    # find path, and candidate
    path, index = find_s_mnp(chunked_sentences, sent_tree_index[:])
    candidate = breadth_first_search(chunked_sentences, path[-1], sent_tree_index[len(index):])
    
    while not valid(candidate, pronoun):
        # Step 4
        if len(index) == 1:
            if index[0] == 0: break
            index = (index[0]-1,)
            candidate = breadth_first_search(chunked_sentences, index, get_first = True)
            if valid(candidate, pronoun): break
            continue

        # Step 5
        path, index = find_s_mnp(chunked_sentences, index)

        # Step 6
        if traverse(chunked_sentences, index)[0].label() == "MNP":
            candidate = breadth_first_search(chunked_sentences, index, get_first = True)
            if valid(candidate, pronoun): break

        # Step 7
        candidate = breadth_first_search(chunked_sentences, index, get_first = True)
        if valid(candidate, pronoun): break

        # Step 8
        if traverse(chunked_sentences, index)[0].label() == "S":
            candidate = breadth_first_search(chunked_sentences, index, right = True)
            if valid(candidate, pronoun): break

    print("hobbs candidate:", candidate)
    return candidate


def extract(chunked_sentences, chunked_indexes):
    """Extract information from chunked_sentences, and determine result. 

    Args:
        chunked_sentences (Tree): result by function 'chunk'
        chunked_indexes (List of Tuples): result by function 'chunk'

    Returns:
        result (Tuple of Booleans): boolean whether names in indexes
            are coreferences of pronoun in indexes.
    """
    sent_index, tree_index, length = chunked_indexes[0]

    # omit length since it's always 1
    coreference = hobbs_algorithm(chunked_sentences, (sent_index, *tree_index))

    # find name_a, name_b
    name_a, name_b = "", ""
    sent_index_a, tree_index_a, length_a = chunked_indexes[1]
    sent_index_b, tree_index_b, length_b = chunked_indexes[2]
    name_a = traverse(chunked_sentences[sent_index_a], tree_index_a, length_a)
    name_b = traverse(chunked_sentences[sent_index_b], tree_index_b, length_b)

    def get_text(t):
        words = []
        for elem in t:
            try:
                elem.label()
                words.append(get_text(elem))
            except AttributeError:
                words.append(elem[0])
        return join(words)
    
    def join(words):
        answer = ""
        prev_was_open_braket = False
        for idx, word in enumerate(words):
            if word in [')', ',', '.', "'s"]:
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

    if not coreference: return (False, False)
    coref = get_text(coreference)
    result_a = get_text(name_a) in coref
    result_b = get_text(name_b) in coref
    if not result_a and not result_b: return (False, True)
    return (result_a, result_b)


def save(mode, result):
    """Build result tsv file.

    Args:
        mode (String): one of snippet, page.
    
    Creates:
        Saves output file in tsv format. 
        Three columns separated by '\t'.
    """
    f = open("CS372_HW5_%s_output_20170305.tsv" % mode, 'w')
    for idx, element in enumerate(result):
        left = "TRUE" if element[0] else "FALSE"
        right = "TRUE" if element[1] else "FALSE"
        content = ["test-%d" % (idx+1), left, right]
        f.write("\t".join(content) + "\n")
    f.close()


def main():
    # get GAP datasets
    development, test, validation = parse_gap()

    # get results
    snippet_results = []
    page_results = []

    # page_contexts
    f = open("CS372_HW5_local_file_1_20170305.tsv", 'r')
    page_contexts = f.read().split('\t\t\t')
    page_contexts = page_contexts[:615] + ["".join(page_contexts[615:617])] + page_contexts[617:]
    f.close()

    for test_idx, item in enumerate(test):
        # # save page contexts
        # page_text = get_page_context(url)
        # page_contexts.add(page_text if page_text else "")
        # continue

        # annotate the snippet
        sentences, indexes, answer, url = annotate_snippet(item)
        chunked_sentences, chunked_indexes = chunk(sentences, indexes)

        # # assert indexes to return same words as given pronoun, name A and B
        # print("text:", item[0])
        # print(item[1] + ",", item[3] + ",", item[6])
        # for sent, word, length in indexes:
        #     print(sentences[sent][word:word+length])
        # for sent, tree, length in chunked_indexes:
        #     cs = chunked_sentences[sent]
        #     for idx, num in enumerate(tree):
        #         if idx == len(tree) - 1:
        #             print(cs[num:num+length])
        #         else:
        #             cs = cs[num]

        result = extract(chunked_sentences, chunked_indexes)
        snippet_results.append(result)

        # print("Answer:", answer)
        # print("(%d) Correct" % test_idx if result == answer else "(%d) Wrong" % test_idx)

        # adjust result with page context
        page_text = page_contexts[test_idx]
        original_text = item[0]
        if not page_text or original_text not in page_text:
            page_results.append(result)
            continue

        # find original text and get neighbour texts.
        page_sentences, updated_indexes = update_annotation(page_text, original_text, indexes)
        # check if invalid..
        invalid = False
        for sent in sentences:
            if not sent in page_sentences:
                invalid = True
        if invalid:
            page_results.append(result)
            continue
        chunked_sentences, chunked_indexes = chunk(page_sentences, updated_indexes)
        result = extract(chunked_sentences, chunked_indexes)
        page_results.append(result)

    # save snippet, page results
    save("snippet", snippet_results)
    save("page", page_results)

    # # save page contexts
    # f = open("page-context.tsv", 'w')
    # f.write("\t\t\t".join(page_contexts) + "\n")
    # f.close()


if __name__ == "__main__":
    main()
