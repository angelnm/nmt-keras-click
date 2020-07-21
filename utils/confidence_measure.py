import sys
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
			prob = self.get_confidence(wordS_source, word_t)
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
		confidence = 1.0

		for word_t in words_target:
			confidence *= self.get_confidence(words_source, word_t)

		len_sentence = len(words_target)
		confidence = confidence ** (1./len_sentence)
		return confidence

	def get_confidence(self, words_source, word_target):
		"""
		Get the confidence measure of a word in a sentence
		:param wordS_source: List of words of the source sentence
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

