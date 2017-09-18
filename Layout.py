import math
import string
import sys
import time

from nltk import FreqDist, re
from nltk.corpus import brown

NWords = 500

def create_index(input_file):
	position = {}
	with open(input_file, 'rb') as file:
		for line in file:
			wordlist = line.split()
			x = int(wordlist[0])
			y = int(wordlist[1])
			distance = int(wordlist[2])
			for word in wordlist[3:]:
				position[word.decode("ascii")] = [x, y]
				x += distance
	return position


class Layout(object):
	def __init__(self, layout_file):
		self.position = create_index(layout_file)

	def calc_clarity(self, allWords):
		nearests_sum = 0
		count = 0
		for word in allWords:
			print(word)
			nearests_sum += self.calc_nearest_neighbour(word, allWords)
			count += 1
		return nearests_sum / count

	def calc_nearest_neighbour(self, candidate, allWords):
		min_dist = 1000000
		for word in allWords:
			if word is candidate:
				continue
			dist = self.calc_ideal_distance(word, candidate)
			min_dist = min(min_dist, dist)
		return min_dist

	def calc_ideal_distance(self, word, candidate):
		dist_sum = 0
		length = min(len(word), len(candidate))
		for i in range(0, length):
			dist_sum += self.distance_between(word[i], candidate[i])
		return dist_sum

	def distance_between(self, word_char, candidate_char):
		c1 = word_char.lower()
		c2 = candidate_char.lower()
		dx = self.position[c1][0] - self.position[c2][0]
		dy = self.position[c1][1] - self.position[c2][1]

		return math.sqrt(dx * dx + dy * dy)


def is_valid(word):
	single_letter = len(word) <= 1
	punctuations = re.search('[' + string.punctuation + ']+', word)
	number = re.search(r'\d', word)
	return not (single_letter or number or punctuations)


def get_word_freq_list():
	words = [word for word in brown.words() if is_valid(word)]
	words_freq_list = FreqDist(i.lower() for i in words).most_common(NWords)
	return words_freq_list


def main():
	layout1 = Layout('layout1')
	topWords = [word for word, frequency in get_word_freq_list()]
	tic = time.clock()
	clarity = layout1.calc_clarity(topWords)
	toc = time.clock()
	print(clarity)
	print(toc - tic)

if __name__ == '__main__':
	sys.exit(main())
