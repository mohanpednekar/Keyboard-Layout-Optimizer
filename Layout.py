import math
import os
import string
import sys
import time
from datetime import datetime
from random import choice

import matplotlib.pyplot as plt
import numpy as np
from nltk import FreqDist, re
from nltk.corpus import brown
from scipy.interpolate import splev

NWords = 5000
MinDistLimit = 8
UnitDistance = 4 * math.sqrt(2)
NParts = 40
DuplicateLetterDist = 4

line_length = 25
line = '=' * line_length
tick = u'\u2713'
dot = u'\u00b7'

summary = {}
begin = time.clock()


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
        alist = [x.rstrip() for x in f]
    while len(picked) < n:
        word = choice(alist)
        if word not in picked:
            picked.append(word)
    return picked


def inline_print(text):
    print(text, end='', flush=True)
    return text


def correction_factor(word):
    return pow(len(word) / NParts, 1 - 0.469)


def speed_for(duration):
    return 10e6 / duration


def duration_for(dist):
    return 68.8 * math.pow(dist, 0.469)


def calc_bspline_duration_for(bs):
    return sum(duration_for(distance_between(bs[i], bs[i + 1])) for i in range(NParts - 1))


def increment_if_better(new, old):
    return inline_print(tick if new < old else dot) is tick


def find_point_at(k1, p1, p2, d):
    x1, y1 = p1
    x2, y2 = p2
    k2 = d - k1
    xp = (k1 * x2 + k2 * x1) / d
    yp = (k1 * y2 + k2 * y1) / d
    p = xp, yp
    return p


def print_results(clarity, speed, runtime):
    print(line)
    print('Clarity    \t= %d' % clarity + '%')
    print('Speed      \t= %d' % speed + '%')
    print('Time Taken \t= %.2f' % runtime + ' sec')
    print(line)


def find_next_anchor(i, k, pos):
    i += 1
    if i is len(pos) - 1: return i
    d = distance_between(pos[i], pos[i + 1])
    while k > d:
        k -= d
        i += 1
        if i is len(pos) - 1: return i - 1
        d = distance_between(pos[i], pos[i + 1])
    if d > 0:
        pos[i] = find_point_at(k, pos[i], pos[i + 1], d)
    return i


class Layout(object):
    def __init__(self, layout_file):
        self.ngrams_division = {}
        self.straight_lines_division = {}
        self.position = {}
        self.bSpline = {}
        self.bSpline_length = {}
        self.straight_length = {}
        self.top_words = []

        with open('layouts/' + layout_file, 'rb') as file:
            self.layout_name = layout_file
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
        count = 0
        for word in self.top_words:
            with open(words_with_length(len(word)), 'a') as file:
                file.write(word + '\n')
            self.bSpline[word] = self.bSpline_for(word)
            self.bSpline_length[word] = self.bSpline_length_for(word)
            self.straight_length[word] = self.straight_length_for(word)
            self.straight_lines_division[word] = self.straight_lines_division_for(word)
            self.ngrams_division[word] = self.nGrams_division_for(word)
            count += 1
            inline_print(dot if count % int(NWords * 0.26) is 0 else '')

    def bSpline_length_for(self, word):
        if word not in self.bSpline: self.bSpline[word] = self.bSpline_for(word)
        return sum(distance_between(self.bSpline[word][i], self.bSpline[word][i + 1]) for i in range(NParts - 1))

    def calc_nearest_neighbour(self, candidate):
        min_dist = MinDistLimit
        nearest = candidate
        for word in self.top_words:
            dist = self.calc_ideal_distance(word, candidate)
            if dist < 1: continue
            min_dist = min(min_dist, dist)
            if min_dist is dist:
                nearest = word
        return nearest

    def calc_ideal_distance(self, word, candidate):
        dist_sum = self.check_first_and_last(candidate, word)
        if dist_sum is MinDistLimit: return MinDistLimit

        wbs = self.bSpline[word]
        cbs = self.bSpline[candidate]

        dist_sum += sum(distance_between(wbs[i], cbs[i]) for i in range(1, NParts))
        return dist_sum / NParts

    def calc_old_distance(self, word, candidate):
        dist_sum = self.check_first_and_last(candidate, word)
        if dist_sum is MinDistLimit: return MinDistLimit

        wsl = self.straight_lines_division[word]
        cbs = self.bSpline[candidate]

        dist_sum += sum(distance_between(wsl[i], cbs[i]) for i in range(1, NParts))
        return dist_sum / NParts

    def calc_new_distance(self, word, candidate):
        dist_sum = self.check_first_and_last(candidate, word)
        if dist_sum is MinDistLimit: return MinDistLimit

        wng = self.ngrams_division[word]
        cbs = self.bSpline[candidate]
        dist_sum += sum(distance_between(wng[i], cbs[i]) for i in range(1, NParts))

        return dist_sum / NParts

    def check_first_and_last(self, candidate, word):
        first = distance_between(self.position[word[0]], self.position[candidate[0]])
        last = distance_between(self.position[word[-1]], self.position[candidate[-1]])
        dist_sum = MinDistLimit if first > UnitDistance or last > UnitDistance else first + last
        return dist_sum

    def distance_between_chars(self, word_char, candidate_char):
        c1 = word_char.lower()
        c2 = candidate_char.lower()
        dx = self.position[c1][0] - self.position[c2][0]
        dy = self.position[c1][1] - self.position[c2][1]
        return math.sqrt(dx * dx + dy * dy)

    def bSpline_for(self, word):
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
        word_count = 0
        count = 0
        for word in picked:
            neighbour = self.calc_nearest_neighbour(word)

            new_distance = self.calc_new_distance(word, neighbour) * 100
            old_distance = self.calc_old_distance(word, neighbour) * 100

            # print(str(int(new_distance)) + '\t' + str(int(old_distance)))
            count += increment_if_better(new_distance, old_distance)

            word_count += 1
            if word_count % line_length is 0: print()
        return count

    def count_better_speed(self, picked):
        count = 0
        word_count = 0
        for word in picked:
            bs = self.bSpline[word]
            duration = calc_bspline_duration_for(bs) * correction_factor(word)
            ideal_speed = speed_for(duration)
            new_speed = self.calc_speed_using_nGrams(word)
            old_speed = self.calc_word_speed_for(word)

            count += increment_if_better(math.fabs(ideal_speed - new_speed), math.fabs(ideal_speed - old_speed))

            word_count += 1
            if word_count % line_length is 0: print()
        return count

    def calc_word_speed_for(self, word):
        duration = 0
        for i in range(len(word) - 1):
            d = self.distance_between_chars(word[i], word[i + 1])
            dist = max(d, DuplicateLetterDist)
            duration += duration_for(dist)
        return speed_for(duration)

    def calc_speed_using_nGrams(self, word):
        max_speed = self.calc_word_speed_for(word)
        for n in range(3, 5):
            with open('ngrams/' + str(n) + 'grams') as ngrams:
                for ngram in ngrams:
                    duration = self.calc_nGram_assisted_duration(ngram, word)
                    if duration is not 0:
                        duration *= correction_factor(word)
                        speed = speed_for(duration)
                        max_speed = max(max_speed, speed)
        return max_speed

    def calc_nGram_assisted_duration(self, ngram, word):
        split_word = re.split("(" + ngram.strip() + ")", word)
        split_word = list(filter(lambda a: a != '', split_word))
        duration = 0
        if len(split_word) > 1:
            for s in split_word:
                if len(s) > 2:
                    duration += calc_bspline_duration_for(self.bSpline_for(s))
                else:
                    if len(s) is 2:
                        duration += duration_for(self.distance_between_chars(s[0], s[1]))
            for i in range(len(split_word) - 1):
                c1 = split_word[i][-1]
                c2 = split_word[i + 1][0]
                duration += duration_for(self.distance_between_chars(c1, c2))
        return duration

    def run_tests(self, picked):
        tic = time.clock()
        clarity = self.count_better_clarity(picked)
        print(line)
        speed = self.count_better_speed(picked)
        toc = time.clock()
        print_results(clarity, speed, toc - tic)
        return [clarity, speed, int(100 * (toc - tic)) / 100]

    def evaluate(self):
        results = {}
        for length in range(4, 11):
            print('\n' + line + '\n\t Word length = ' + str(length) + '\n' + line)
            results[length] = self.run_tests(pick(length))
        summary[self.layout_name] = results

    def straight_length_for(self, word):
        return sum(self.distance_between_chars(word[i], word[i + 1]) for i in range(len(word) - 1))

    def straight_lines_division_for(self, word):
        k = self.straight_length[word] / NParts
        pos = {}
        for i in range(len(word)):
            pos[i] = self.position[word[i]]
        points = np.empty([0, 2])
        points = np.append(points, [pos[0]], 0)
        points = self.divide_using_points_list(k, points, pos)
        points = np.append(points, [pos[len(word) - 1]], 0)
        return points

    def divide_using_points_list(self, k, points, pos):
        i = 0
        while i < len(pos) - 1:
            d = distance_between(pos[i], pos[i + 1])
            while d > k:
                pos[i] = find_point_at(k, pos[i], pos[i + 1], d)
                points = np.append(points, [pos[i]], 0)
                d -= k
            i = find_next_anchor(i, k - d, pos)
            points = np.append(points, [pos[i]], 0)
        return points

    def nGrams_division_for(self, word):
        min_length = self.straight_length[word]
        best_ngram = word
        for n in range(3, 5):
            with open('ngrams/' + str(n) + 'grams') as ngrams:
                for ngram in ngrams:
                    length = self.calc_nGram_assisted_length(ngram, word)
                    if length < min_length:
                        min_length = length
                        best_ngram = ngram
        if best_ngram is not word:
            return self.calc_vector_for(word, best_ngram)
        return self.straight_lines_division[word]

    def calc_vector_for(self, word, ngram):
        split_word = re.split("(" + ngram.strip() + ")", word)
        split_word = list(filter(lambda a: a != '', split_word))
        part_length = self.calc_nGram_assisted_length(ngram, word) / NParts
        points = np.empty([0, 2])
        posarray = []
        for s in split_word:
            if s is ngram:
                posarray.extend(self.bSpline_for(s))
            else:
                posarray.extend([self.position[ch] for ch in s])
        pos = {}
        for i in range(len(posarray)):
            pos[i] = posarray[i]

        points = np.append(points, [pos[0]], 0)
        points = self.divide_using_points_list(part_length, points, pos)
        points = np.append(points, [posarray[- 1]], 0)

        return points

    def calc_nGram_assisted_length(self, ngram, word):
        split_word = re.split("(" + ngram.strip() + ")", word)
        split_word = list(filter(lambda a: a != '', split_word))

        length = 0
        if len(split_word) > 1:
            for s in split_word:
                if len(s) > 2:
                    length += self.bSpline_length_for(s)
                else:
                    if len(s) is 2:
                        length += self.distance_between_chars(s[0], s[1])
            for i in range(len(split_word) - 1):
                c1 = split_word[i][-1]
                c2 = split_word[i + 1][0]
                length += self.distance_between_chars(c1, c2)
        else:
            length = self.straight_length[word]
        return length


def is_valid(word):
    single_letter = len(word) <= 1
    punctuations = re.search('[' + string.punctuation + ']+', word)
    number = re.search(r'\d', word)
    return not (single_letter or number or punctuations)


def get_top_words():
    inline_print(dot)
    words = [word for word in brown.words() if is_valid(word)]
    inline_print(dot)
    words_freq_list = FreqDist(i.lower() for i in words).most_common(NWords)
    inline_print(dot)
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


def print_final_results(results):
    plt.figure(figsize=(12, 6), dpi=100)
    end = time.clock()
    x = np.arange(4, 11)
    count = 0
    with open('results.txt', 'a+') as file:
        file.write('{:%Y-%m-%d %H:%M:%S}\t'.format(datetime.now()))
        file.write(str(int(end - begin)) + ' seconds\n')

        for key, value in results.items():
            file.write(str(key))
            file.write('\n')
            file.write(str(value))
            file.write('\n')

            count += 1
            ax = plt.subplot(1, 2, count)
            plt.ylim(0, 100)

            c = [value[i][0] for i in x]
            s = [value[i][1] for i in x]
            t = [value[i][2] for i in x]

            ax.plot(x, c)
            ax.plot(x, s)
            ax.plot(x, t)

            ax.legend(['Clarity', 'Speed', 'Time'], loc='upper left')
            ax.set_title(key)

        file.write('\n')
    print("Summary is in results.txt\n")
    plt.tight_layout()
    plt.savefig('result.png')


def main():
    inline_print('\nGetting top words\t')
    top_words = get_top_words()
    print(' ' + tick)

    layout_files = os.listdir('layouts')
    layout_files.sort()
    for layout_name in layout_files:
        inline_print('Training ' + layout_name + '\t')
        layout = Layout(layout_name)
        layout.train(top_words)
        print(' ' + tick)
        layout.evaluate()
        print()

    print_final_results(summary)


def test():
    print('Uh oh.. This is test().\nChange last line to run main()')

if __name__ == '__main__':
    sys.exit(main() if 1 else test())
