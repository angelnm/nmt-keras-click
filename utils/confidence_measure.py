import sys
import math
import numpy as np
from scipy.sparse import csr_matrix, load_npz

class CM:
	DIC_EXT = 'txt'
	MAT_EXT = 'npz'
	END_P = '</SEG>'
	SOURCE = 0
	TARGET = 1

	def __init__(self, model_path, verbose=False):
		self.verbose = verbose

		self.probability_matrix = self.load_matrix(model_path)
		self.nonzero_matrix = self.load_nonzero_lexicon()
		self.lexicon_smoothing = 0.00

		self.words2index, self.index2words = self.load_dictionaries(model_path)

	def log(self, value):
		if value != 0:
			return math.log(value)
		return -math.inf

	def load_dictionaries(self, model_path):
		"""
		Load the dictionaries to translate words to indices and vice versa
		:param model_path: Path to the file
		:return: words2index & index2words dictionaries
		"""
		try:
			with open(f"{model_path}.{CM.DIC_EXT}", 'r') as f:
				lines = f.read().splitlines()

			words2index = [{CM.END_P:0}, {CM.END_P:0}]
			index2words = [{0:CM.END_P}, {0:CM.END_P}]

			idx = 1
			for word in lines[0].split()[1:]:
				index2words[CM.SOURCE][idx] = word
				words2index[CM.SOURCE][word] = idx
				idx += 1

			idx = 1
			for word in lines[1].split()[1:]:
				index2words[CM.TARGET][idx] = word
				words2index[CM.TARGET][word] = idx
				idx += 1

			return words2index, index2words

		except FileNotFoundError:
			file_not_found_alert('{}.{}'.format(model_path, CM.DIC_EXT))

	def load_matrix(self, model_path):
		"""
		Load the matrix of probabilities
		:param model_path: Path to the file
		:return: csr_matrix with the probabilities [Source x Target]
		"""
		try:
			return load_npz(f"{model_path}.{CM.MAT_EXT}")
		except FileNotFoundError:
			file_not_found_alert('{}.{}'.format(model_path, CM.MAT_EXT))

	def load_nonzero_lexicon(self):
		shape = self.probability_matrix.shape

		nonzero = []
		for idx in range(shape[0]):
			nz = self.probability_matrix[idx, :].count_nonzero()
			nonzero.append(shape[1]-nz)

		return nonzero

	def get_ratio_confidence(self, words_source, words_target, threshold):
		"""
		Calculate the ration confidence measure of a sentence
		:param words_source: List of words from the source sentence
		:param words_target: List of words from the target sentence
		:param threshold: Threshold to mark a word as correct
		:return: Confidence measure
		"""
		correct_words = 0.0

		for word_t in words_target:
			prob = self.get_confidence(words_source, word_t, pos+1, len(words_target))
			if prob >= threshold:
				correct_words += 1

		len_sentence = len(words_target)
		confidence = correct_words / len_sentence

		return confidence

	def get_mean_confidence(self, words_source, words_target):
		"""
		Calculate the mean confidence measure of a sentence
		:param words_source: List of words from the source sentence
		:param words_target: List of words form the target sentence
		:return: Confidence measure 
		"""
		confidence = 0.0

		for pos, word_t in enumerate(words_target):
			value = self.log(self.get_confidence(words_source, word_t, pos+1, len(words_target)))
			confidence += value

		len_sentence = len(words_target)
		confidence = confidence/len_sentence
		confidence = math.exp(confidence)

		return confidence

	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		pass

	def get_lexicon_probability(self, word_source, word_target):
		"""
		Get the lexion probability
		:param word_source: Word from the source sentence
		:param word_target: Word from the target sentence
		:return: Probability of the target being the translation of source
		"""
		try:
			idx_source = self.words2index[CM.SOURCE][word_source]
		except Exception:
			self.no_word_alert(word_source)
			return 0.0

		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		prob = self.probability_matrix[idx_source, idx_target]
		if prob == 0:
			prob = self.lexicon_smoothing / self.nonzero_matrix[idx_source]
		else:
			prob *= (1 - self.lexicon_smoothing)

		return prob

	def no_word_alert(self, word):
		if self.verbose:
			print(f"'{word}' isn't in the dictionary")

	def file_not_found_alert(self, file_path):
		print('{} not found!'.format(file_path))
		sys.exit()

class IBM1(CM):

	def __init__(self, model_path, verbose=False):
		CM.__init__(self, model_path, verbose)

	def get_confidence(self, sentence_source, word_target, target_pos=None, target_len=None):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:return: Max translate probability
		"""
		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		max_prob = 0.0
		for word_source in sentence_source:
			prob = self.get_lexicon_probability(word_source, word_target)

			if prob > max_prob:
				max_prob = prob

		return max_prob

class IBM2(CM):

	def __init__(self, model_path, alignment_path, verbose=False):
		CM.__init__(self, model_path, verbose)
		self.alignment_matrix = self.load_alignment(alignment_path)

	def load_alignment(self, alignment_path):
		"""
		Load the alignment probabilities
		:param alignment_path: Path to the file
		:return: alignment_matrix
		"""
		try:
			alignment_matrix = dict()
			with open(alignment_path, 'r') as f:
				for line in f:
					elements = line.split()
					s_pos = int(elements[0])
					t_pos = int(elements[1])
					s_length = int(elements[2])
					t_length = int(elements[3])
					probability = float(elements[4])

					if s_length not in alignment_matrix:
						alignment_matrix[s_length] = dict()
					if t_length not in alignment_matrix[s_length]:
						new_table = np.zeros((s_length+1, t_length+1))
						alignment_matrix[s_length][t_length] = new_table

					current_table = alignment_matrix[s_length][t_length]
					current_table[s_pos][t_pos] = probability

			return alignment_matrix	    
		except FileNotFoundError:
			file_not_found_alert(alignment_path)

	def get_alignment_probability(self, s_pos, t_pos, s_len, t_len):
		"""
		Get the probability of the current alignment
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: Probability of the alignment
		"""
		if s_len in self.alignment_matrix and t_len in self.alignment_matrix[s_len]:
			return self.alignment_matrix[s_len][t_len][s_pos][t_pos]
		else:
			return 0.0

	def get_confidence(self, sentence_source, word_target, target_pos, target_len):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param target_pos: Position of the target word
		:param target_len: Length of the target sentence
		:return: Max translate probability
		"""
		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		source_len = len(sentence_source)

		max_prob = self.log(self.get_alignment_probability(0, target_pos, source_len, target_len)) + self.log(self.get_lexicon_probability(CM.END_P, word_target))
		for pos, word in enumerate(sentence_source):
			prob = self.log(self.get_alignment_probability(pos+1, target_pos, source_len, target_len)) + self.log(self.get_lexicon_probability(word, word_target))
			if prob > max_prob:
				max_prob = prob

		return math.exp(max_prob)

class Fast_Align(CM):

	def __init__(self, model_path, prob_0 = 0, tension = 4, verbose=False):
		CM.__init__(self, model_path, verbose)

		self.prob_0 = prob_0
		self.tension = tension

	def get_alignment_probability(self, s_pos, t_pos, s_len, t_len, norm_factor):
		"""
		Get the probability of the current alignment
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:param norm_factor: Normalization factor to use the fast_align equation
		:return: Probability of the alignment
		"""
		if s_pos == 0:
			return self.prob_0
		elif s_pos <= s_len:
			foo = self.get_e(s_pos, t_pos, s_len, t_len)
			foo /= norm_factor
			foo *= (1 - self.prob_0)
			return foo
		else:
			return 0.0

	def get_h(self, s_pos, t_pos, s_len, t_len):
		"""
		Implementation of the equation: h(i,j,m,n) = -|frac{i}{m}-frac{j}{n}|"
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: -|frac{t_pos}{t_len}-frac{s_pos}{s_len}|
		"""
		a = t_pos/t_len
		b = s_pos/s_len

		a -= b
		a = -abs(a)

		return a

	def get_e(self, s_pos, t_pos, s_len, t_len):
		"""
		Implementation of the equation: e^{lambda h(i,j,m,n)}
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: e^{self.tension h(t_pos,s_pos,t_len,s_len)}
		"""
		a = self.get_h(s_pos, t_pos, s_len, t_len)
		a *= self.tension
		a = math.exp(a)
		return a

	def get_s(self, s_pos, t_pos, s_len, t_len, r, l):
		"""
		Implementation of the equation: s_l(g,r) = g frac{1-r^l}{1-r}
		:param s_pos: Position of the source word
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:param r:
		:param l:
		:return: e(t_pos, s_pos, t_len, s_len) frac{1-r^l}{1-r}
		"""
		g = self.get_e(s_pos, t_pos, s_len, t_len)

		a = 1 - math.pow(r, l)
		b = 1 - r
		a /= b

		return g*a

	def get_normalize_factor(self, t_pos, s_len, t_len):
		"""
		Calculate the normalization factor for the current target position
		Equation: s_{j1}(e^{lambda h(i,j1,m,n)}, r) + s_{n-j2}(e^{lambda h(i,j2,m,n)}, r)
		:param t_pos: Position of the target word
		:param s_len: Length of the source sentence
		:param t_len: Length of the target sentence
		:return: Normalization factor
		"""
		j_up = math.floor((t_pos/t_len)*s_len)
		j_dw = j_up + 1

		r = math.exp(-self.tension/s_len)

		s_up = self.get_s(j_up, t_pos, s_len, t_len, r, j_up)
		s_dw = self.get_s(j_dw, t_pos, s_len, t_len, r, s_len-j_dw+1)

		return s_up + s_dw

	def get_confidence(self, sentence_source, word_target, target_pos, target_len):
		"""
		Get the confidence measure of a word in a sentence
		:param sentence_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param target_pos: Position of the target word
		:param target_len: Length of the target sentence
		:return: Max Translate Probability
		"""
		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		source_len = len(sentence_source)
		norm_factor = self.get_normalize_factor(target_pos, source_len, target_len)

		max_prob = self.log(self.get_alignment_probability(0, target_pos, source_len, target_len, norm_factor)) + self.log(self.get_lexicon_probability(CM.END_P, word_target))
		for pos, word in enumerate(sentence_source):
			prob = self.log(self.get_alignment_probability(pos+1, target_pos, source_len, target_len, norm_factor)) + self.log(self.get_lexicon_probability(word, word_target))

			if prob > max_prob:
				max_prob = prob

		return math.exp(max_prob)