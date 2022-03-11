# Natural Language Processing

This is a repository for coursework in Natural Language Processing in CS372, 2020 Spring, KAIST.

## About the Course and NLTK

Please refer to the link : [Course](http://nlpcl.kaist.ac.kr/~cs372_2020/index.php), [NLTK Book](http://www.nltk.org/book/)

## About the Coursework

### HW1

Find pairs of expressions that are 'intensity-modifying'.

For example,

> Extol, praise _highly_  
> Destitute, _very_ poor

### HW2

Find pairs of words where one word is both 'restricting' and 'intensity-modifying' another word.

For example,

> _pitch_ black  
> _dead_ center  
> _deathly_ sick  
> _stark_ contrast

### HW3

Find sentences that has most occurrence of heteronyms.  
Single occurrence of heteronym is also counted. 

* Homograph: Same letters, different meaning.  
* Heteronym: Among homographs, different pronounciation. 

For example, 

> contains: _'wind(air)'_ + _'wind(tie)'_ + _'tear(pull apart)'_ + _'tear(droplet)'_   
> contains: _'wind(air)'_ + _'wind(tie)'_ + _'tear(pull apart)'_  
> ...

### HW4

Extract relations from [MEDLINE database](https://pubmed.ncbi.nlm.nih.gov/). *<X, ACTION, Y>*  
Collect 100 sentences with annotated relations.  
Use 80 sentences to train, test module with remaining 20 sentences. 

For example, 

> Inorganic phosphate inhibited HPr kinase but activated HPR phosphatase.  
> => <Inorganic phosphate, inhibited, HPr kinase>  
> => <Inorganic phosphate, activated, HPR phosphatase>

> All vasodilators activated K-Cl cotransport in LK SRBCs and HYZ in VSMCs, and this activation was inhibited by calyculin and genistein, two inhibitors of K-Cl cotransport.  
> => <All vasodilators, activated, K-CI cotransport>  
> => <All vasodilators, activated, HYZ>  
> => <this activation, was inhibited by, calyculin> OR <calyculin, inhibited, this activation>  
> => <this activation, was inhibited by, genistein> OR <genistein, inhibited, this activation>  
> => <this activation, was inhibited by, two inhibitors> OR <two inhibitors, inhibited, this activation>

### HW5

Resolve coreferences from [GAP coreference dataset](https://github.com/google-research-datasets/gap-coreference).  
Find right antecedent names for ambiguous pronouns for each snippet.   
Divide searching algorithm into two: snippet-context(only uses given text), and page-context(uses whole context from Wikipedia URL).

For example,

> In May, *Fujisawa* joined *Mari Motohashi’s* rink as the team’s skip, moving back from Karuizawa to Kitami where **she** had spent her junior days.  
> => ‘Fujisawa’ is ‘TRUE’ and ‘Mari Motohashi’ is ‘FALSE’.

## Environments

Refer to how to setup [here](./setup).

## Other References

> [NLTK Wordnet](https://frhyme.github.io/python-lib/nltk-wordnet/)  
> [Part-of-speech Tagging](https://medium.com/@muddaprince456/categorizing-and-pos-tagging-with-nltk-python-28f2bc9312c3)

