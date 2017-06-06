import nltk
import os

# takes a file path as argument
# returns list of sentences, each represented as a list of tokens
def tokenize(path) :
	sentences = []
	words = []
	file = open(path, "r", encoding = "utf-8")
	text = file.read().replace("\n", " ")
	file.close()
	sentences = nltk.sent_tokenize(text)
	words = []
	for sentence in sentences :
		words.append(nltk.word_tokenize(sentence))
	return words

# represents an ngram model
class Ngram :
	def __init__(self, n) :
		# model is stored as a dictionary of dictionaries
		# outer keys are the first n-1 elements of the ngram (the predictive part)
		# inner keys are the last elements of the respective ngrams (the predicted part)
		# the inner values are the frequencies
		self.__model = {}
		self.__n = n - 1 # this is a shortcut for splitting input more easily later on
		self.__types = set([]) # because for 3+ grams len(__model) doesn't necessarily give the number of distinct tokens
	
	# takes a sentence as argument
	# updates the model with the new ngrams
	def update(self, sentence) :
		# add new types to type set
		self.__types |= set(sentence)
	
		# add sentence begin and end tags
		sentence = ["^^^"] + sentence + ["$$$"]
		
		# find all n-1 grams except the last
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			
			# check if n-1 gram and full ngram are recognized, if so increment count by one
			if key in self.__model and sentence[i] in self.__model[key] :
				self.__model[key][sentence[i]] += 1
			else :
				# if even the n-1 gram is not recognized, expand the outer dictionary
				if key not in self.__model :
					self.__model[key] = {}
				
				self.__model[key][sentence[i]] = 1
	
	# takes a sentence as argument
	# evaluates likelihood (MLE) of the sentence against the model
	def prob_mle(self, sentence) :
		# probability of sentence so far
		p = 1.0
		
		sentence = ["^^^"] + sentence + ["$$$"] # cf. update()
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			if key not in self.__model or sentence[i] not in self.__model[key]:
				# if the ngram isn't in the model its MLE probability is 0, which propagates and makes everything 0 anyway
				return 0
			
			# the sum of all the possibilities given the first n-1 terms so the conditional probablility can be calculated
			total = sum([self.__model[key][w] for w in self.__model[key]])
			
			# total probability is the product of the conditional probabilities
			p *= self.__model[key][sentence[i]] / total
		return p
	
	# takes a sentence as argument
	# optionally takes alpha as argument, defaults to 1 (Laplace)
	# evaluates smoothed likelihood of the sentence against the model
	def prob_add(self, sentence, alpha = 1) :
		# probability of sentence so far
		p = 1.0
		
		sentence = ["^^^"] + sentence + ["$$$"] # cf. update()
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			C = alpha # alpha + occurrences of the target ngram
			if key in self.__model and sentence[i] in self.__model[key]:
				C += self.__model[key][sentence[i]]
			
			# the sum of all the possibilities given the first n-1 terms so the conditional probablility can be calculated
			N = sum([self.__model[key][w] for w in self.__model[key]])
			V = len(self.__types)
			
			# total probability is the product of the conditional probabilities
			p *= self.__model[key][sentence[i]] / (N + (alpha * V))
		return p
	
	def __str__(self) :
		return "Order " + str(self.__n + 1) + " ngram model with " + self.__token_ct + " entries:\n[" + str(self.__model) + "]"

bigrams = Ngram(2)
for file in os.listdir("./assignment1-data/") :
	if file[0] == 'c' :
		print(file)
		for sentence in tokenize("./assignment1-data/" + file) :
			bigrams.update(sentence)

sent = ["I", "had", "."]
print(bigrams.prob_mle(sent))
print(bigrams.prob_add(sent))
print(bigrams.prob_add(sent, 0.5))