from PeterSchoenerAlonBorenstein import *

sentences = [["This", "is", "a", "test", "."], ["I", "'m", "sorry", "Dave", "."], ["I", "'m", "afraid", "I", "can", "'t", "do", "that", "."]]

test_model = BackoffNgram(3)

for sentence in sentences :
	test_model.update(sentence)

print(test_model.prob(["This", "is", "a", "test", "."]))
print(test_model.prob(["This", "is", "Dave", "."]))
