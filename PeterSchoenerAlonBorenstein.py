import nltk
#--------------Ex1------------------
def tokenize(path):
    sentences=[]
    words=[]
    for line in open(path,"r",encoding="utf-8"):
        sentences.append(nltk.sent_tokenize(line))
        #for word in line:
        #    words.append(nltk.word_tokenize(
    print(sentences)
tokenize("C:/Users/alon464/Desktop/assignment1-data/c00.txt")
print("hi Alon")
