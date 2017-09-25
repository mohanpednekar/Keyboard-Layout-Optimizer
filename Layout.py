import math
import os
import string
import sys
import time
from random import choice

import numpy as np
from nltk import FreqDist, re
from nltk.corpus import brown
from scipy.interpolate import splev

NWords = 5000
MinDistLimit = 4
UnitDistance = 4 * math.sqrt(2) + 2
NParts = 40
DuplicateLetterDist = 4

line = '=' * 25
tick = u'\u2713'
dot = u'\u00b7'


def distance_between(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	dx = x1 - x2
	dy = y1 - y2
	return math.sqrt(dx * dx + dy * dy)


def words_with_length(i):
	return 'len/len' + str(i)


def pick(nChars, n=100):
	picked = []
	with open(words_with_length(nChars)) as f:
		alist = [line.rstrip() for line in f]
	while len(picked) < n:
		word = choice(alist)
		if word not in picked:
			picked.append(word)
	return picked


def inline_print(text):
	print(text, end='', flush=True)


def correction_factor(word):
	return pow(len(word) / NParts, 1 - 0.469)


def speed_for(duration):
	return 10e6 / duration


def duration_for(dist):
	return 68.8 * math.pow(dist, 0.469)


def calc_bspline_duration_for(bs):
	duration = 0
	for i in range(NParts - 1):
		dist = distance_between(bs[i], bs[i + 1])
		duration += duration_for(dist)
	return duration


class Layout(object):
	def __init__(self, layout_file):
		self.position = {}
		self.bSpline = {}
		self.bSpline_double_precision = {}
		self.length = {}
		self.good_words_count = 0
		self.top_words = []

		with open('layouts/' + layout_file, 'rb') as file:
			for line in file:
				wordList = line.split()
				x = int(wordList[0])
				y = int(wordList[1])
				distance = int(wordList[2])
				for word in wordList[3:]:
					self.position[word.decode("ascii")] = [x, y]
					x += distance

	def train(self, top_words):
		self.top_words = top_words
		for i in range(20):
			filename = words_with_length(i)
			if os.path.exists(filename):
				os.remove(filename)
		for word in self.top_words:
			with open(words_with_length(len(word)), 'a') as file:
				file.write(word + '\n')
			self.bSpline[word] = self.bSpline_for(word)
			self.bSpline_double_precision[word] = self.bSpline_double_precision_for(word)
			self.length[word] = 0
			for i in range(NParts - 1):
				self.length[word] += distance_between(self.bSpline[word][i], self.bSpline[word][i + 1])

	def calc_nearest_neighbour(self, candidate):
		min_dist = MinDistLimit
		for word in self.top_words:
			dist = self.calc_distance(word, candidate)
			if dist < 1: continue
			min_dist = min(min_dist, dist)
		return min_dist

	def calc_distance(self, word, candidate):
		word_bspline = self.bSpline[word]
		candidate_bspline = self.bSpline[candidate]

		dist_sum = distance_between(word_bspline[0], candidate_bspline[0])
		if dist_sum > UnitDistance:
			return MinDistLimit
		last = distance_between(word_bspline[-1], candidate_bspline[-1])
		if last > UnitDistance:
			return MinDistLimit
		dist_sum += last

		for i in range(1, NParts):
			dist_sum += distance_between(word_bspline[i], candidate_bspline[i])
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

	def bSpline_for(self, word):
		cv = np.empty([0, 2])
		for c in word:
			cv = np.append(cv, [self.position[c]], 0)
		return bSpline(cv)

	def bSpline_double_precision_for(self, word):
		pos = {}
		for i in range(len(word)):
			pos[2 * i] = self.position[word[i]]
		for i in range(len(word) - 1):
			pos[2 * i + 1] = [0.5 * (pos[2 * i][0] + pos[2 * i + 2][0]), 0.5 * (pos[2 * i][1] + pos[2 * i + 2][1])]
		cv = np.empty([0, 2])
		for i in range(len(word) - 1):
			cv = np.append(cv, [pos[i]], 0)
		return bSpline(cv)

	def count_better_clarity(self, picked):
		nearests_sum = 0
		word_count = 0
		for word in picked:
			min_dist = self.calc_nearest_neighbour(word)
			if min_dist is MinDistLimit:
				inline_print(tick)
				self.good_words_count += 1
			else:
				inline_print(dot)
			nearests_sum += min_dist
			word_count += 1
			if word_count % 25 is 0: print()
		return nearests_sum / word_count

	def count_better_speed(self, picked):
		count = 0
		word_count = 0
		for word in picked:

			bs = self.bSpline_double_precision[word]
			duration = calc_bspline_duration_for(bs) * correction_factor(word)
			ideal_speed = speed_for(duration)

			new_speed = self.check_ngram(word)

			old_speed = self.calc_word_speed_for(word)

			if math.fabs(ideal_speed - old_speed) > math.fabs(ideal_speed - new_speed):
				inline_print(tick)
				count += 1
			else:
				inline_print(dot)
			word_count += 1
			if word_count % 25 is 0: print()
		return count

	def calc_word_speed_for(self, word):
		duration = 0
		for i in range(len(word) - 1):
			d = self.distance_between_chars(word[i], word[i + 1])
			dist = max(d, DuplicateLetterDist)
			duration += duration_for(dist)
		return speed_for(duration)

	def check_ngram(self, word):
		max_speed = self.calc_word_speed_for(word)
		for n in range(3, 5):
			with open('ngrams/' + str(n) + 'grams') as ngrams:
				for ngram in ngrams:
					splitted = re.split("(" + ngram.strip() + ")", word)
					splitted = list(filter(lambda a: a != '', splitted))
					duration = 0
					if len(splitted) > 1:
						for s in splitted:
							if len(s) > 2:
								bs = self.bSpline_double_precision_for(s)
								duration += calc_bspline_duration_for(bs)
							else:
								if len(s) is 2:
									duration += duration_for(self.distance_between_chars(s[0], s[1]))
						for i in range(len(splitted) - 1):
							c1 = splitted[i][-1]
							c2 = splitted[i + 1][0]
							duration += duration_for(self.distance_between_chars(c1, c2))
					if duration is 0: continue
					duration *= correction_factor(word)
					speed = speed_for(duration)
					max_speed = max(max_speed, speed)
		return max_speed

	def run_tests(self, picked):
		tic = time.clock()
		self.good_words_count = 0
		clarity = self.count_better_clarity(picked)
		print(line)
		speed = self.count_better_speed(picked)

		toc = time.clock()
		self.print_results(clarity, speed, toc - tic)

	def print_results(self, clarity, speed, runtime):
		print(line)
		print('Good Words \t= %d' % self.good_words_count)
		print('Clarity    \t= %d' % clarity + '%')
		print('Speed      \t= %d' % speed + '%')
		print('Time Taken \t= %.2f' % runtime + ' sec')
		print(line)

	def evaluate(self):
		for length in range(4, 11):
			print('\n' + line + '\n\t Word length = ' + str(length) + '\n' + line)
			self.run_tests(pick(length))


def is_valid(word):
	single_letter = len(word) <= 1
	punctuations = re.search('[' + string.punctuation + ']+', word)
	number = re.search(r'\d', word)
	return not (single_letter or number or punctuations)


def get_top_words():
	words = [word for word in brown.words() if is_valid(word)]
	words_freq_list = FreqDist(i.lower() for i in words).most_common(NWords)
	top_words = [word for word, frequency in words_freq_list]
	with open('misc/' + 'topWords', 'w+') as file:
		for word in top_words: file.write(word + '\n')
	return top_words


def bSpline(cv, n=NParts, degree=2):
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
	inline_print('\nGetting top words\t... ')
	top_words = get_top_words()
	print(tick)

	layout_files = ['layout1']
	for layout_name in layout_files:
		inline_print('Training ' + layout_name + '\t... ')
		layout = Layout(layout_name)
		layout.train(top_words)
		print(tick)
		layout.evaluate()


if __name__ == '__main__':
	sys.exit(main())
