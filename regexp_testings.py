import nltk.tokenize as nt
import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer


def extract_NN(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>}

    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne

def extract_keywords(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>|<JJ>*<NN>|<NN>*<JJ>|<NN><VB.*><JJ>|<VBG>|<VBN>}
    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne


text = input('text: ')
# text = 'distention of abdomen'
# # text = text.lower()
while True:
    ss=nt.sent_tokenize(text)
    tokenized_sent=[nt.word_tokenize(sent) for sent in ss]
    sentence_word = list()
    for sentence in tokenized_sent:
        sentence_word.append( [lemmatizer.lemmatize(word) for word in sentence] )

    print(sentence_word)

    pos_sentences=[nltk.pos_tag(sent) for sent in sentence_word]
    print('\nPos_Sentences: ',pos_sentences,'\n')
    print(extract_keywords(text))
    text = input('text: ')

# ww = list()
# for i in sentence_word:
#     ww = ww +' '+ i 
# print('untokenized words: ',sentence_word)



# # print(extract_NN(text))

