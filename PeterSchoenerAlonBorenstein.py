# TO UPDATE FILE:
# git commit -m "message"
# git push
# TO GET NEW VERSION
# git pull

import nltk
#--------------Ex1------------------
def tokenize(path):
    sentences=[]
    words=[]
    file=open(path,"r",encoding="utf-8")
    text=file.read().replace("\n", " ")
    sentences = nltk.sent_tokenize(text)
    words = []
    for sentence in sentences :
        words.append(nltk.word_tokenize(sentence))
    print(words)
tokenize("C:/Users/alon464/Desktop/assignment1-data/c00.txt")
print("hi Alon")
