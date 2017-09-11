import sys


def main():
	pass


if __name__ == '__main__':
	sys.exit(main())


def create_index(file):
	pass


class Layout(object):
	def __init__(self, layout_file):
		self.position = create_index(layout_file)

	def calc_clarity(self, allWords):
		nearests_sum = 0
		count = 0
		for word in allWords:
			nearests_sum += self.calc_nearest_neighbour(word, allWords)
			count += 1
		return nearests_sum/count

	def calc_nearest_neighbour(self, candidate, allWords):
		min_dist = 1000000
		for word in allWords:
			dist = self.calc_ideal_distance(word, candidate)
			min_dist = min(min_dist, dist)
		return min_dist

	def calc_ideal_distance(self, word, candidate):
		dist_sum = 0
		for i in range(0, word.length):
			dist_sum += self.distance_between(word[i], candidate[i])
		return dist_sum

	def distance_between(self, word_char, candidate_char):
		w_int = int(word_char-'a')
		c_int = int(candidate_char-'b')
		d = self.position[w_int-c_int]
		return d*d
