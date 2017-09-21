import math
import string
import sys
import time

import numpy as np
from nltk import FreqDist, re
from nltk.corpus import brown
from scipy.interpolate import splev

NWords = 5000
MinDistLimit = 8
UnitDistance = 4 * math.sqrt(2) + 2
NParts = 40


class Layout(object):
	def __init__(self, layout_file):
		self.position = {}
		self.bSpline = {}
		self.length = {}
		self.good_words_count = 0
		with open(layout_file, 'rb') as file:
			for line in file:
				wordList = line.split()
				x = int(wordList[0])
				y = int(wordList[1])
				distance = int(wordList[2])
				for word in wordList[3:]:
					self.position[word.decode("ascii")] = [x, y]
					x += distance

	def train(self, allWords):
		for word in allWords:
			self.bSpline[word] = self.bspline_for(word)
			self.length[word] = 0
			for i in range(NParts - 1):
				self.length[word] += self.distance_between(self.bSpline[word][i], self.bSpline[word][i + 1])

	def calc_clarity(self, allWords):
		nearests_sum = 0
		count = 0

		for word in allWords:
			min_dist = self.calc_nearest_neighbour(word, allWords)
			if min_dist is MinDistLimit:
				print(word)
				self.good_words_count += 1
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
			dist = self.calc_distance(word, candidate)
			min_dist = min(min_dist, dist)
		return min_dist

	def calc_distance(self, word, candidate):
		dist_sum = self.distance_between_chars(word[0], candidate[0])
		if dist_sum > UnitDistance:
			return MinDistLimit
		last = self.distance_between_chars(word[-1], candidate[-1])
		if last > UnitDistance:
			return MinDistLimit
		dist_sum += last

		word_bspline = self.bSpline[word]
		candidate_bspline = self.bSpline[candidate]
		for i in range(1, NParts):
			dist_sum += self.distance_between(word_bspline[i], candidate_bspline[i])
		return dist_sum / NParts

	def calc_ideal_distance(self, word, candidate):
		dist_sum = self.distance_between_chars(word[0], candidate[0])
		if dist_sum > UnitDistance:
			return MinDistLimit
		last = self.distance_between_chars(word[-1], candidate[-1])
		if last > UnitDistance:
			return MinDistLimit
		dist_sum += last
		for i in range(1, len(word) - 1):
			dist_sum += self.distance_between_chars(word[i], candidate[i])
		return dist_sum

	def distance_between_chars(self, word_char, candidate_char):
		c1 = word_char.lower()
		c2 = candidate_char.lower()
		dx = self.position[c1][0] - self.position[c2][0]
		dy = self.position[c1][1] - self.position[c2][1]

		return math.sqrt(dx * dx + dy * dy)

	def distance_between(self, p1, p2):
		x1, y1 = p1
		x2, y2 = p2

		dx = x1 - x2
		dy = y1 - y2
		return math.sqrt(dx * dx + dy * dy)

	def bspline_for(self, word):
		cv = np.empty([0, 2])
		for c in word:
			cv = np.append(cv, [self.position[c]], 0)
		return bspline(cv)

	def calc_speed(self, topWords):
		total_duration_per_letter = 0
		for word in topWords:
			duration = 0
			for i in range(len(word) - 1):
				duration += 68.8 * math.pow(self.distance_between_chars(word[i], word[i + 1]), 0.469)
			total_duration_per_letter += duration / len(word)
		return int(total_duration_per_letter) / NWords


def is_valid(word):
	single_letter = len(word) <= 1
	punctuations = re.search('[' + string.punctuation + ']+', word)
	number = re.search(r'\d', word)
	return not (single_letter or number or punctuations)


def get_word_freq_list():
	words = [word for word in brown.words() if is_valid(word)]
	words_freq_list = FreqDist(i.lower() for i in words).most_common(NWords)
	return words_freq_list


def bspline(cv, n=NParts, degree=2):
	cv = np.asarray(cv)
	count = len(cv)

	degree = np.clip(degree, 1, count - 1)
	kv = np.array([0] * degree + list(range(count - degree + 1)) + [count - degree] * degree, dtype='int')
	u = np.linspace(False, (count - degree), n)
	arange = np.arange(len(u))
	points = np.zeros((len(u), cv.shape[1]))
	for i in range(cv.shape[1]):
		points[arange, i] = splev(u, (kv, cv[:, i], degree))
	return points


def main():
	topWords = [word for word, frequency in get_word_freq_list()]
	with open('topWords.txt', 'w') as topWordsFile:
		for word in topWords:
			topWordsFile.write(word)

	tic = time.clock()
	layout1 = Layout('layout1')
	layout1.train(topWords)
	clarity = layout1.calc_clarity(topWords)
	speed = layout1.calc_speed(topWords)
	toc = time.clock()
	print('-' * 24)
	print('Clarity \t= ' + str(int(clarity * 100) / MinDistLimit) + ' %')
	print('Good Words \t= ' + str(layout1.good_words_count * 100 / NWords) + ' %')
	print('Speed \t\t= ' + str(speed))
	print('Time \t\t= ' + str(int(toc - tic)) + ' sec')
	print('-' * 24)


def test():
	pass


if __name__ == '__main__':
	sys.exit(main())
