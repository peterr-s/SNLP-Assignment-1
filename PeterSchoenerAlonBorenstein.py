import nltk

#--------------Ex1------------------
def tokenize(path) :
    sentences = []
    words = []
    file = open(path, "r", encoding = "utf-8")
    text = file.read().replace("\n", " ")
    sentences = nltk.sent_tokenize(text)
    words = []
    for sentence in sentences :
        words.append(nltk.word_tokenize(sentence))
    print(words)

tokenize("./assignment1-data/c00.txt")