import nltk
import os

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
	return words

class Ngram :
	def __init__(self, n) :
		self.model = {}
		self.n = n - 1
	
	def update(self, sentence) :
		sentence = ["^^^"] + sentence + ["$$$"]
		for i in range(self.n, len(sentence)) :
			key = tuple(sentence[i - self.n : i])
			if key in self.model and sentence[i] in self.model[key] :
				self.model[key][sentence[i]] += 1
			else :
				if key not in self.model :
					self.model[key] = {}
				self.model[key][sentence[i]] = 1
	
	def prob_mle(self, sentence) :
		p = 1.0
		sentence = ["^^^"] + sentence + ["$$$"]
		for i in range(self.n, len(sentence)) :
			key = tuple(sentence[i - self.n : i])
			if key not in self.model :
				return 0
			total = sum([self.model[key][w] for w in self.model[key]])
			p *= self.model[key][sentence[i]] / total
		return p
	
	def printyourself(self) :
		print(self.model)

bigrams = Ngram(2)
for file in os.listdir("./assignment1-data/") :
	if file[0] == 'c' :
		print(file)
		for sentence in tokenize("./assignment1-data/" + file) :
			bigrams.update(sentence)

print(bigrams.prob_mle(["I", "had", "."]))