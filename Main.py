import itertools
import string
import sys

from nltk import FreqDist, re
from nltk.corpus import brown

NWords = 5000
NGCount = 100


def get_probability(freq, ng, i):
	c = chr(i + 96)
	candidate_freq = Ngram.all_n_grams.count(ng + c)
	prob = candidate_freq / freq
	return prob


class Ngram(object):
	candidates = []

	def __init__(self, ng, n):
		self.ng = ng
		self.n = n
		self.freq = Ngram.all_n_grams.count(ng)
		prob_sum = 0
		for i in range(1, 27):
			prob = get_probability(self.freq + 1, self.ng, i)
			prob_sum += prob
			self.candidates.append(prob)
		self.candidates.insert(0, 1 - prob_sum)

	def __str__(self):
		return self.ng


Ngram.all_n_grams = None


def get_n_grams(words_freq_list, n):
	ng = [n_grams_for(word, n) for word, freq in words_freq_list]
	ng = list(itertools.chain.from_iterable(ng))
	Ngram.all_n_grams = ng
	print("\nTotal = " + ng.__len__().__str__())
	ng = FreqDist(ng)
	print("Unique = " + ng.__len__().__str__())
	print_word_freq(ng, NGCount, n)
	return ng


def main():
	words_freq_list = get_word_freq_list()
	n_grams = [get_n_grams(words_freq_list, i) for i in range(2, 5)]


def get_co_ordinates(word):
	pass


def is_valid(word):
	single_letter = len(word) <= 1
	punctuations = re.search('[' + string.punctuation + ']+', word)
	number = word.isnumeric()
	return not (single_letter or number or punctuations)


def get_word_freq_list():
	words = [word for word in brown.words() if is_valid(word)]
	words_freq_list = FreqDist(i.lower() for i in words).most_common(NWords)
	return words_freq_list


def print_word_freq(freq_list, count, n):
	with open('ngrams/' + str(n) + 'grams', 'w+') as file:
		for word, frequency in freq_list.most_common(count):
			print(u'{} {}'.format(word, frequency))
			file.write(word + '\n')


def n_grams_for(word, n):
	n -= 1
	return [word[i - n:i + 1] for i, char in enumerate(word)][n:]


def test():
	pass


if __name__ == '__main__':
	sys.exit(main())
