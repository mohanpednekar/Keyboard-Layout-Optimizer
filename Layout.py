import math
import string
import sys
import time

from nltk import FreqDist, re
from nltk.corpus import brown
from scipy.interpolate import BSpline

NWords = 5000
MinDistLimit = 200
UnitDistance = 4 * math.sqrt(2) + 2


class Layout(object):
	def __init__(self, layout_file):
		self.position = {}
		with open(layout_file, 'rb') as file:
			for line in file:
				wordList = line.split()
				x = int(wordList[0])
				y = int(wordList[1])
				distance = int(wordList[2])
				for word in wordList[3:]:
					self.position[word.decode("ascii")] = [x, y]
					x += distance

	def calc_clarity(self, allWords):
		nearests_sum = 0
		count = 0
		for word in allWords:
			min_dist = self.calc_nearest_neighbour(word, allWords)
			if min_dist is MinDistLimit:
				print(word)
			nearests_sum += min_dist
			count += 1
		return nearests_sum / count

	def calc_nearest_neighbour(self, candidate, allWords):
		min_dist = MinDistLimit
		for word in allWords:
			if len(word) is not len(candidate):
				continue
			if word is candidate:
				continue
			dist = self.calc_ideal_distance(word, candidate)
			min_dist = min(min_dist, dist)
		return min_dist

	def calc_ideal_distance(self, word, candidate):
		dist_sum = self.distance_between(word[0], candidate[0])
		if dist_sum > UnitDistance:
			return MinDistLimit
		last = self.distance_between(word[-1], candidate[-1])
		if last > UnitDistance:
			return MinDistLimit
		dist_sum += last
		for i in range(1, len(word) - 1):
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


def printBSpline(word):
	pass


def main():
	topWords = [word for word, frequency in get_word_freq_list()]
	with open('topWords.txt', 'w') as topWordsFile:
		for word in topWords:
			topWordsFile.write(word)
			printBSpline(word)

	tic = time.clock()
	layout1 = Layout('layout1')
	clarity = layout1.calc_clarity(topWords)
	toc = time.clock()

	print(clarity)
	print(toc - tic)


def test():
	t = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20]
	c = [-1, 2, 0, -1, 2, 3, 4, 5, 6, 7, 8]
	k = 3
	spl = BSpline(t, c, k, False)
	for i in range(7):
		print(spl(i))


if __name__ == '__main__':
	sys.exit(test())
