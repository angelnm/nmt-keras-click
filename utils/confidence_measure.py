import sys
import math
from scipy.sparse import csr_matrix, load_npz

class CM1:
	DIC_EXT = 'txt'
	MAT_EXT = 'npz'
	END_P = '</SEG>'
	SOURCE = 0
	TARGET = 1

	def __init__(self, model_path, verbose=False):
		self.verbose = verbose

		self.probability_matrix = self.load_matrix(model_path)
		self.words2index, self.index2words = self.load_dictionaries(model_path)

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
			print(f"{model_path}.{CM.DIC_EXT} not found!")
			sys.exit()


	def load_matrix(self, model_path):
		"""
		Load the matrix of probabilities
		:param model_path: Path to the file
		:return: csr_matrix with the probabilities [Source x Target]
		"""
		try:
			return load_npz(f"{model_path}.{CM.MAT_EXT}")
		except FileNotFoundError:
			print(f"{model_path}.{CM.MAT_EXT} not found!")
			sys.exit()

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
			prob = self.get_confidence(words_source, word_t)
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

		for word_t in words_target:
			value = math.log(max(self.get_confidence(words_source, word_t), 1e-10))
			confidence += value

		len_sentence = len(words_target)
		confidence = confidence/len_sentence
		confidence = math.exp(confidence)

		return confidence

	def get_confidence(self, words_source, word_target):
		"""
		Get the confidence measure of a word in a sentence
		:param words_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:return: Max translate probability
		"""
		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			if self.verbose:
				print(f"'{word_target}' isn't in the dictionary")
			return 0.0

		max_prob = 0.0
		for word in words_source:
			try:
				idx_source = self.words2index[CM.SOURCE][word]
				prob = self.probability_matrix[idx_source, idx_target]
			except Exception:
				prob = 0.0

			if prob > max_prob:
				max_prob = prob

		return max_prob

	def get_probability(self, word_source, word_target):
		"""
		Get the translate probability of both words
		:param word_source: Word to translate
		:param word_target: Translated word
		:return: Probabilty of the translation
		"""
		try:
			idx_source = self.words2index[CM.SOURCE][word_source]
		except Exception:
			if self.verbose:
				print(f"'{word_source}' isn't in the dictionary")
			return 0.0

		try:
			idx_target = self.words2index[CM.TARGET][word_target]
		except Exception:
			if self.verbose:
				print(f"'{word_target}' isn't in the dictionary")
			return 0.0

		return self.probability_matrix[idx_source, idx_target]

	def max_probability(self, word_source):
		"""
		Get the max translate probability word
		:param word_source: Word to translate
		:return: Translated word
		"""
		try:
			idx_source = self.words2index[CM.SOURCE][word_source]
		except Exception:
			if self.verbose:
				print(f"'{word_source}' isn't in the dictionary")
			return None

		idx_max = self.probability_matrix[idx_source, :].argmax()
		return self.index2words[CM.TARGET][idx_max]

class CM2:
	DIC_EXT = 'txt'
	MAT_EXT = 'npz'
	END_P = '</SEG>'
	SOURCE = 0
	TARGET = 1

	def __init__(self, model_path, alignment_path, verbose=False):
		self.verbose = verbose

		self.alignment_matrix = self.load_alignment(alignment_path)
		self.probability_matrix = self.load_matrix(model_path)
		self.words2index, self.index2words = self.load_dictionaries(model_path)

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
			print(f"{alignment_path} not found!")
			sys.exit()

	def load_dictionaries(self, model_path):
		"""
		Load the dictionaries to translate words to indices and vice versa
		:param model_path: Path to the file
		:return: words2index & index2words dictionaries
		"""
		try:
			with open(f"{model_path}.{CM2.DIC_EXT}", 'r') as f:
				lines = f.read().splitlines()

			words2index = [{CM2.END_P:0}, {CM2.END_P:0}]
			index2words = [{0:CM2.END_P}, {0:CM2.END_P}]

			idx = 1
			for word in lines[0].split()[1:]:
				index2words[CM2.SOURCE][idx] = word
				words2index[CM2.SOURCE][word] = idx
				idx += 1

			idx = 1
			for word in lines[1].split()[1:]:
				index2words[CM2.TARGET][idx] = word
				words2index[CM2.TARGET][word] = idx
				idx += 1

			return words2index, index2words

		except FileNotFoundError:
			print(f"{model_path}.{CM2.DIC_EXT} not found!")
			sys.exit()


	def load_matrix(self, model_path):
		"""
		Load the matrix of probabilities
		:param model_path: Path to the file
		:return: csr_matrix with the probabilities [Source x Target]
		"""
		try:
			return load_npz(f"{model_path}.{CM2.MAT_EXT}")
		except FileNotFoundError:
			print(f"{model_path}.{CM2.MAT_EXT} not found!")
			sys.exit()

	def get_ratio_confidence(self, words_source, words_target, threshold):
		"""
		Calculate the ration confidence measure of a sentence
		:param words_source: List of words from the source sentence
		:param words_target: List of words from the target sentence
		:param threshold: Threshold to mark a word as correct
		:return: Confidence measure
		"""
		correct_words = 0.0

		for pos, word_t in enumerate(words_target):
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

		for word_t in words_target:
			value = math.log(max(self.get_confidence(words_source, word_t), 1e-10))
			confidence += value

		len_sentence = len(words_target)
		confidence = confidence/len_sentence
		confidence = math.exp(confidence)

		return confidence

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

	def get_lexicon_probability(self, word_source, word_target):
		"""
		Get the lexion probability
		:param word_source: Word from the source sentence
		:param word_target: Word from the target sentence
		:return: Probability of the target being the translation of source
		"""
		try:
			idx_source = self.words2index[CM2.SOURCE][word_source]
		except Exception:
			self.no_word_alert(word_source)
			return 0.0

		try:
			idx_target = self.words2index[CM2.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		return self.probability_matrix[idx_source, idx_target]

	def get_confidence(self, words_source, word_target, target_pos, target_len):
		"""
		Get the confidence measure of a word in a sentence
		:param wordS_source: List of words of the source sentence
		:param word_target: Word from the target sentence
		:param target_pos: Position of the target word
		:param target_len: Length of the target sentence
		:return: Max translate probability
		"""
		try:
			idx_target = self.words2index[CM2.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		max_prob = self.get_alignment_probability(0, target_pos, len(words_source), target_len) * self.get_lexicon_probability(CM2.END_P, word_target)
		for pos, word in enumerate(words_source):
			prob = self.get_alignment_probability(pos+1, target_pos, len(words_source), target_len) * self.get_lexicon_probability(word, word_target)

			if prob > max_prob:
				max_prob = prob

		return max_prob

	def get_probability(self, word_source, word_target):
		"""
		Get the translate probability of both words
		:param word_source: Word to translate
		:param word_target: Translated word
		:return: Probabilty of the translation
		"""
		try:
			idx_source = self.words2index[CM2.SOURCE][word_source]
		except Exception:
			self.no_word_alert(word_source)
			return 0.0

		try:
			idx_target = self.words2index[CM2.TARGET][word_target]
		except Exception:
			self.no_word_alert(word_target)
			return 0.0

		return self.probability_matrix[idx_source, idx_target]

	def max_probability(self, word_source):
		"""
		Get the max translate probability word
		:param word_source: Word to translate
		:return: Translated word
		"""
		try:
			idx_source = self.words2index[CM2.SOURCE][word_source]
		except Exception:
			self.no_word_alert(word_source)
			return None

		idx_max = self.probability_matrix[idx_source, :].argmax()
		return self.index2words[CM2.TARGET][idx_max]

	# =======================================================================================================

	def no_word_alert(word):
		if self.verbose:
			print(f"'{word}' isn't in the dictionary")

