import nltk
import os
import math
import matplotlib.pyplot

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
		self.N = 0
	
	# takes a sentence as argument
	# updates the model with the new ngrams
	def update(self, sentence) :
		# add new types to type set
		self.__types |= set(sentence)
	
		# add sentence begin and end tags
		sentence = ["^^^"] * self.__n + sentence + ["$$$"] * self.__n
		
		# find all n-1 grams except the last
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			
			# check if n-1 gram and full ngram are recognized, if so increment count by one
			if key in self.__model and sentence[i] in self.__model[key] :
				self.__model[key][sentence[i]] += 1
				self.__model[key]["%%%TOTAL"] += 1
			else :
				# if even the n-1 gram is not recognized, expand the outer dictionary
				if key not in self.__model :
					self.__model[key] = {"%%%TOTAL":0}
				
				self.__model[key][sentence[i]] = 1
				self.__model[key]["%%%TOTAL"] += 1
		self.N = sum([self.__model[k]["%%%TOTAL"] for k in self.__model])
	
	# takes a sentence as argument
	# evaluates likelihood (MLE) of the sentence against the model
	def prob_mle(self, sentence) :
		# probability of sentence so far
		p = 1.0
		
		sentence = ["^^^"] * self.__n + sentence + ["$$$"] * self.__n # cf. update()
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			if key not in self.__model or sentence[i] not in self.__model[key]:
				# if the ngram isn't in the model its MLE probability is 0, which propagates and makes everything 0 anyway
				return 0
			
			# the sum of all the possibilities given the first n-1 terms so the conditional probablility can be calculated
			total = self.__model[key]["%%%TOTAL"]#sum([self.__model[key][w] for w in self.__model[key]])
			
			# total probability is the product of the conditional probabilities
			p *= self.__model[key][sentence[i]] / total
		return p
	
	def __log_prob_add(self, sentence, alpha = 1) :
		# probability of sentence so far
		p = 0
		
		sentence = ["^^^"] * self.__n + sentence + ["$$$"] * self.__n # cf. update()
		for i in range(self.__n, len(sentence)) :
			key = tuple(sentence[i - self.__n : i])
			C = alpha # alpha + occurrences of the target ngram
			if key in self.__model and sentence[i] in self.__model[key]:
				C += self.__model[key][sentence[i]]
			# the sum of all the possibilities given the first n-1 terms so the conditional probablility can be calculated
			#N = 0
			#if key in self.__model :
			#	N = self.__model[key]["%%%TOTAL"]
			V = len(self.__types)
			
			# total probability is the product of the conditional probabilities
			p += math.log2(C / (self.N + (alpha * V))) # this prevents underflow (reversed in return statement)
		
	#	print(p)
		return p
	
	# takes a sentence as argument
	# optionally takes alpha as argument, defaults to 1 (Laplace)
	# evaluates smoothed likelihood of the sentence against the model
	def prob_add(self, sentence, alpha = 1) :
		return pow(2, self.__log_prob_add(sentence, alpha))
	
	def perplexity(self, sentences, alpha = 1) :
		# calculate product of sentence likelihoods
		p = 0
		for sentence in sentences :
			p += self.__log_prob_add(sentence, alpha)
		
		h = -1 * p / len(self.__types)
		return pow(2, h)

	def _perplexity_helper(self, alpha) :
		return self.perplexity(self.__temp_sentences, alpha)
	def estimate_alpha(self, sentences, n = 3) :
		self.__temp_sentences = sentences
		a1 = 0.25
		a2 = 0.5
		a3 = 0.75
		
		a2 = min(a1, a2, a3, key=self._perplexity_helper)
		d = 0.125
		a1 = a2 - d
		a3 = a2 + d
		
		for i in range(1, n) :
#			print(a2)
			a2 = min(a1, a2, a3, key=self._perplexity_helper)
			d /= 2
			a1 = a2 - d
			a3 = a2 + d
		
		self.__temp_sentences = []
		return min(a1, a2, a3, key=self._perplexity_helper)

	def __str__(self) :
		return "Order " + str(self.__n + 1) + " ngram model with " + self.__token_ct + " entries:\n[" + str(self.__model) + "]"

bigrams = Ngram(2)
sentences = []
d_sentences =  []
for file in os.listdir("./assignment1-data/")[0:3] + os.listdir("./assignment1-data/")[28:30] :
	if file[0] == 'c' and file[1] == '0' :
		print(file)
		sentences += tokenize("./assignment1-data/" + file)
	elif file[0] == 'd' and file[1] == '0' :
		print(file)
		d_sentences += tokenize("./assignment1-data/" + file)

i = 0
l = len(sentences)
for sentence in sentences :
#	print(sentence)
	bigrams.update(sentence)
	i += 1
	if i % 100 == 0 :
		n = int(i * 50 / l)
		print("\r[", n * '#', (48 - n) * ' ', "]", 2 * n, '%', end='')

print()
a = bigrams.estimate_alpha(d_sentences[0:10], 5)
print(a)
print(bigrams.perplexity(d_sentences[0:100], a))
a = bigrams.estimate_alpha(sentences[0:10], 5)
print(a)
print(bigrams.perplexity(sentences[0:100], a))