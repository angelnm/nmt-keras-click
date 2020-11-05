# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import argparse
import ast
import codecs
import copy
import logging
import time
import pandas as pd
from collections import OrderedDict

from keras_wrapper.model_ensemble import InteractiveBeamSearchSampler
from keras_wrapper.extra.isles_utils import *
from keras_wrapper.online_trainer import OnlineTrainer

from config import load_parameters
from config_online import load_parameters as load_parameters_online
from data_engine.prepare_data import update_dataset_from_file
from keras_wrapper.cnn_model import loadModel, updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.read_write import pkl2dict, list2file
from keras_wrapper.utils import decode_predictions_beam_search, flatten_list_of_lists
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.online_models import build_online_models
from utils.utils import update_parameters
from utils.confidence_measure import CM, IBM1, IBM2, Fast_Align
from sys import version_info

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.ter.ter import Ter


logging.basicConfig(level=logging.DEBUG,
					format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(2)

def check_params(parameters):
	assert parameters['BEAM_SEARCH'], 'Only beam search is supported.'

def parse_args():
	parser = argparse.ArgumentParser("Simulate an interactive NMT session")
	parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
	parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'],
						help="Splits to sample. Should be already included into the dataset object.")
	parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
	parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
															   "If not specified, hyperparameters "
															   "are read from config.py")
	parser.add_argument("--max-n", type=int, default=3, help="Maximum number of words generated between isles")
	parser.add_argument("-src", "--source", help="File of source hypothesis", required=True)
	parser.add_argument("-trg", "--references", help="Reference sentence (for simulation)", required=True)
	parser.add_argument("-bpe-tok", "--tokenize-bpe", help="Apply BPE tokenization", action='store_true', default=True)
	parser.add_argument("-bpe-detok", "--detokenize-bpe", help="Revert BPE tokenization",
						action='store_true', default=True)
	parser.add_argument("-tok-ref", "--tokenize-references", help="Tokenize references. If set to False, the references"
																  "should be given already preprocessed/tokenized.",
						action='store_true', default=False)
	parser.add_argument("-d", "--dest", required=True, help="File to save translations in")
	parser.add_argument("-od", "--original-dest", help="Save original hypotheses to this file", required=False)
	parser.add_argument("-p", "--prefix", action="store_true", default=False, help="Prefix-based post-edition")
	parser.add_argument("-o", "--online",
						action='store_true', default=False, required=False,
						help="Online training mode after postedition. ")
	parser.add_argument("--models", nargs='+', required=True, help="path to the models")
	parser.add_argument("-ch", "--changes", nargs="*", help="Changes to config, following the syntax Key=Value",
						default="")
	parser.add_argument("-ma", "--max_mouse_actions", 	 type=int, 	 required=False, default=0, 	help="Max number of mouse actions for the same position")

	parser.add_argument("-est","--confidence_estimator", type=int,	 required=False, default=0, 	help="Confidence Estimator to use. 0-IBM1 | 1-IBM2 | 2-Fast_Align")
	parser.add_argument("-lm", "--lexicon_model", 		 type=str, 	 required=True,  				help="path to the model of the confidence measure")
	parser.add_argument("-am", "--alignment_model",  	 type=str, 	 required=True,  				help="path to the alignment model of the confidence measure")
	#parser.add_argument("Mean or Ratio")
	parser.add_argument("-st", "--sentence_threshold", 	 type=float, required=False, default=1.0, 	help="Sentence threshold")
	#parser.add_argument("Ratio threshold")
	parser.add_argument("-wt", "--word_threshold", 		 type=int, 	 required=False, default=2, 	help="Words threshold")

	return parser.parse_args()

def export_csv(output_path, data):
	# threshold, missclassified, tag_correct, tag_incorrect, CER
	df = pd.DataFrame(data)
	df.to_csv(output_path, index=False)

def get_sentence_cm(confidence_model, reference, hypothesis, method=0, ratio=0.0):
	sentence_cm = 0

	if method == 0:
		sentence_cm = confidence_model.get_mean_confidence(reference, hypothesis)
	elif method == 1:
		sentence_cm = confidence_model.get_ratio_confidence(reference, hypothesis, ratio)

	return sentence_cm

def calculate_scores(scorers, refs, hypos):
	scores_sentence = {}
	for scorer, method in scorers:
		score, _ = scorer.compute_score(refs, hypos)
		if isinstance(score, list):
			for m, s in list(zip(method, score)):
				scores_sentence[m] = s
		else:
			scores_sentence[method] = score

	return scores_sentence

def generate_constrained_hypothesis(beam_searcher, src_seq, fixed_words_user, params, args, isle_indices, filtered_idx2word,
									index2word_y, sources, heuristic, mapping, unk_indices, unk_words, unks_in_isles, unk_id=1):
	"""
	Generates and decodes a user-constrained hypothesis given a source sentence and the user feedback signals.
	:param src_seq: Sequence of indices of the source sentence to translate.
	:param fixed_words_user: Dict of word indices fixed by the user and its corresponding position: {pos: idx_word}
	:param args: Simulation options
	:param isle_indices: Isles fixed by the user. List of (isle_index, [words])
	:param filtered_idx2word: Dictionary of possible words according to the current word prefix.
	:param index2word_y: Indices to words mapping.
	:param sources: Source words (for unk replacement)
	:param heuristic: Unk replacement heuristic
	:param mapping: Source--Target dictionary for Unk replacement strategies 1 and 2
	:param unk_indices: Indices of the hypothesis that contain an unknown word (introduced by the user)
	:param unk_words: Corresponding word for unk_indices
	:return: Constrained hypothesis
	"""
	# Generate a constrained hypothesis
	trans_indices, costs, alphas = beam_searcher. \
		sample_beam_search_interactive(src_seq,
									   fixed_words=copy.copy(fixed_words_user),
									   max_N=args.max_n,
									   isles=isle_indices,
									   valid_next_words=filtered_idx2word,
									   idx2word=index2word_y)

	# Substitute possible unknown words in isles
	unk_in_isles = []
	for isle_idx, isle_sequence, isle_words in unks_in_isles:
		if unk_id in isle_sequence:
			unk_in_isles.append((subfinder(isle_sequence, list(trans_indices)), isle_words))

	if params['pos_unk']:
		alphas = [alphas]
	else:
		alphas = None

	# Decode predictions
	hypothesis = decode_predictions_beam_search([trans_indices],
												index2word_y,
												alphas=alphas,
												x_text=sources,
												heuristic=heuristic,
												mapping=mapping,
												pad_sequences=True,
												verbose=0)[0]
	hypothesis = hypothesis.split()
	for (words_idx, starting_pos), words in unk_in_isles:
		for pos_unk_word, pos_hypothesis in enumerate(range(starting_pos, starting_pos + len(words_idx))):
			hypothesis[pos_hypothesis] = words[pos_unk_word]

	# UNK words management
	if len(unk_indices) > 0:  # If we added some UNK word
		if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
			for i, index in enumerate(range(0, len(hypothesis))):
				hypothesis[index] = unk_words[unk_indices[i]]
			for ii in range(i + 1, len(unk_words)):
				hypothesis.append(unk_words[ii])
		else:  # We put each unknown word in the corresponding gap
			for i, index in enumerate(unk_indices):
				if index < len(hypothesis):
					hypothesis[index] = unk_words[i]
				else:
					hypothesis.append(unk_words[i])

	return hypothesis

def interactive_simulation():
	args = parse_args()

	if args.config is not None:
		logger.info('Reading parameters from %s.' % args.config)
		params = update_parameters({}, pkl2dict(args.config))
	else:
		logger.info('Reading parameters fron config.py.')
		params = load_parameters()

	if args.online:
		online_parameters = load_parameters_online(params)
		params = update_parameters(params, online_parameters)

	try:
		for arg in args.changes:
			try:
				k, v = arg.split('=')
			except ValueError:
				print('Overwritten arguments must have the from key=Value. \n Currently are: %s' % str(args.changes))
				exit(1)
			try:
				params[k] = ast.literal_eval(v)
			except ValueError:
				params[k] = v
	except ValueError:
		print('Error processing arguments: (', k, ',', v, ')')
		exit(2)

	check_params(params)
	if args.verbose:
		logging.info('params = ' + str(params))
	dataset = loadDataset(args.dataset)
	dataset = update_dataset_from_file(dataset, args.source, params, splits=args.splits, remove_outputs=True)

	bpe_separator = dataset.BPE_separator if hasattr(dataset, 'BPE_separator') and dataset.BPE_separator is not None else u'@@'

	params['TOKENIZATION_METHOD'] = 'tokenize_bpe' if args.tokenize_bpe else params.get('TOKENIZATION_METHOD', 'tokenize_none')

	if 'bpe' in params['TOKENIZATION_METHOD'].lower():
		logger.info('Building BPE')
		if not dataset.BPE_built:
			dataset.build_bpe(params.get('BPE_CODES_PATH', params['DATA_ROOT_PATH'] + '/training_codes.joint'),
							separator=bpe_separator)

	tokenize_f = eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none'))

	if args.online:
		params_training = {  # Traning params
			'n_epochs': params['MAX_EPOCH'],
			'shuffle': False,
			'loss': params.get('LOSS', 'categorical_crossentropy'),
			'batch_size': params.get('BATCH_SIZE', 1),
			'homogeneous_batches': False,
			'optimizer': params.get('OPTIMIZER', 'SGD'),
			'lr': params.get('LR', 0.1),
			'lr_decay': params.get('LR_DECAY', None),
			'lr_gamma': params.get('LR_GAMMA', 1.),
			'epochs_for_save': -1,
			'verbose': args.verbose,
			'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
			'n_parallel_loaders': params['PARALLEL_LOADERS'],
			'extra_callbacks': [],  # callbacks,
			'reload_epoch': 0,
			'epoch_offset': 0,
			'data_augmentation': params['DATA_AUGMENTATION'],
			'patience': params.get('PATIENCE', 0),
			'metric_check': params.get('STOP_METRIC', None),
			'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
			'each_n_epochs': params.get('EVAL_EACH', 1),
			'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0),
			'additional_training_settings': {'k': params.get('K', 1),
											 'tau': params.get('TAU', 1),
											 'lambda': params.get('LAMBDA', 0.5),
											 'c': params.get('C', 0.5),
											 'd': params.get('D', 0.5)
											 }
		}
	else:
		params_training = dict()

	params['INPUT_VOCABULARY_SIZE']  = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
	params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
	logger.info('<<< Using an ensemble of %d models >>>' % len(args.models))

	if args.online:
		# Load trainable model(s)
		logging.info('Loading models from %s' % str(args.models))
		model_instances = [TranslationModel(params,
											model_type=params['MODEL_TYPE'],
											verbose=params['VERBOSE'],
											model_name=params['MODEL_NAME'] + '_' + str(i),
											vocabularies=dataset.vocabulary,
											store_path=params['STORE_PATH'],
											clear_dirs=False,
											set_optimizer=False)
						   for i in range(len(args.models))]
		models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, args.models)]

		# Set additional inputs to models if using a custom loss function
		params['USE_CUSTOM_LOSS'] = True if 'PAS' in params['OPTIMIZER'] else False
		if params['N_BEST_OPTIMIZER']:
			logging.info('Using N-best optimizer')

		models = build_online_models(models, params)
		online_trainer = OnlineTrainer(models,
									   dataset,
									   None,
									   None,
									   params_training,
									   verbose=args.verbose)
	else:
		# Otherwise, load regular model(s)
		models = [loadModel(m, -1, full_path=True) for m in args.models] 

	fsrc = codecs.open(args.source, 'r', encoding='utf-8')
	ftrans = codecs.open(args.dest, 'w', encoding='utf-8')
	logger.info('<<< Storing corrected hypotheses into: %s >>>' % str(args.dest))

	if args.original_dest is not None:
		logger.info('<<< Storing original hypotheses into: %s >>>' % str(args.original_dest))
		ftrans_ori = open(args.original_dest, 'w')
		ftrans_ori.close()

	ftrg = codecs.open(args.references, 'r', encoding='utf-8')
	target_lines = ftrg.read().split('\n')
	if target_lines[-1] == u'':
		target_lines = target_lines[:-1]

	index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
	word2index_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['words2idx']
	index2word_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['idx2words']
	word2index_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['words2idx']
	unk_id = dataset.extra_words['<unk>']

	# Load Confidence Measure Model
	if args.confidence_estimator == 0:
		confidence_model = IBM1(args.lexicon_model)
	elif args.confidence_estimator == 1:
		confidence_model = IBM2(args.lexicon_model, args.alignment_model)
	elif args.confidence_estimator == 2:
		confidence_model = Fast_Align(args.lexicon_model)
	else:
		exit()

	sentence_threshold = args.sentence_threshold
	#word_metrics['threshold'] = np.append(np.arange(0.0, 1.0, 1/args.word_threshold), 1.0)

	word_metrics = {}
	word_metrics['threshold'] 		= [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-15, 1e-20, 1e-25, 1e-50, 0.0]
	word_metrics['missclasified']	= [0 for x in word_metrics['threshold']]
	word_metrics['tag_correct'] 	= [0 for x in word_metrics['threshold']]
	word_metrics['tag_incorrect'] 	= [0 for x in word_metrics['threshold']]
	word_metrics['CER']				= [0 for x in word_metrics['threshold']]


	logger.debug(args.word_threshold)
	logger.debug(str(word_metrics['threshold']))

	# Initialize counters
	total_errors = 0
	total_words = 0
	total_chars = 0
	total_mouse_actions = 0
	total_keystrokes = 0
	total_sentences = 0
	total_wrong_sentences = 0
	total_words_checked = 0

	try:
		for s in args.splits:
			params_prediction = {'max_batch_size': params['BATCH_SIZE'],
								 'n_parallel_loaders': params['PARALLEL_LOADERS'],
								 'predict_on_sets': [s],
								 'beam_size': params['BEAM_SIZE'],
								 'maxlen': params['MAX_OUTPUT_TEXT_LEN_TEST'],
								 'optimized_search': params['OPTIMIZED_SEARCH'],
								 'model_inputs': params['INPUTS_IDS_MODEL'],
								 'model_outputs': params['OUTPUTS_IDS_MODEL'],
								 'dataset_inputs': params['INPUTS_IDS_DATASET'],
								 'dataset_outputs': params['OUTPUTS_IDS_DATASET'],
								 'normalize_probs': params['NORMALIZE_SAMPLING'],
								 'alpha_factor': params['ALPHA_FACTOR'],
								 'pos_unk': params['POS_UNK'],
								 'heuristic': params['HEURISTIC'],
								 'search_pruning': params.get('SEARCH_PRUNING', False),
								 'state_below_index': -1,
								 'output_text_index': 0,
								 'apply_tokenization': params.get('APPLY_TOKENIZATION', False),
								 'tokenize_f': eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none')),
								 'apply_detokenization': params.get('APPLY_DETOKENIZATION', True),
								 'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none')),
								 'coverage_penalty': params.get('COVERAGE_PENALTY', False),
								 'length_penalty': params.get('LENGTH_PENALTY', False),
								 'length_norm_factor': params.get('LENGTH_NORM_FACTOR', 0.0),
								 'coverage_norm_factor': params.get('COVERAGE_NORM_FACTOR', 0.0),
								 'state_below_maxlen': -1 if params.get('PAD_ON_BATCH', True) else params.get('MAX_OUTPUT_TEXT_LEN_TEST', 50),
								 'output_max_length_depending_on_x': params.get('MAXLEN_GIVEN_X', True),
								 'output_max_length_depending_on_x_factor': params.get('MAXLEN_GIVEN_X_FACTOR', 3),
								 'output_min_length_depending_on_x': params.get('MINLEN_GIVEN_X', True),
								 'output_min_length_depending_on_x_factor': params.get('MINLEN_GIVEN_X_FACTOR', 2),
								 'attend_on_output': params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower()),
								 'n_best_optimizer': params.get('N_BEST_OPTIMIZER', False)
								 }

			# Manage pos_unk strategies
			if params['POS_UNK']:
				mapping = None if dataset.mapping == dict() else dataset.mapping
			else:
				mapping = None

			# Build interactive samples
			interactive_beam_searcher = InteractiveBeamSearchSampler(models,
																	 dataset,
																	 params_prediction,
																	 excluded_words = None,
																	 verbose=args.verbose)
			start_time = time.time()

			if args.verbose:
				logging.info('Params prediction = ' + str(params_prediction))
				if args.online:
					logging.info('Params training = ' + str(params_training))

			# Set the scorers
			scorers = [
					(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
					#(Ter(), "TER")
				]

			# Variables to calculate the BLEU & TER
			refs_metrics = {}
			hypo_metrics = {}

			# Start to translate the source file interactively
			for n_line, src_line in enumerate(fsrc):
				# Initialize sentence counters
				errors_sentence = 0
				keystrokes_sentence = 0
				mouse_actions_sentence = 0
				hypothesis_number = 0
				unk_indices = []

				# Get (tokenized) input
				tokenized_input = src_line.strip()
				if params_prediction.get('apply_tokenization'):
					tokenized_input = tokenize_f(tokenized_input)

				# Convert text to indices
				src_seq = dataset.loadText([tokenized_input],
										   vocabularies=dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]],
										   max_len=params['MAX_INPUT_TEXT_LEN'],
										   offset=0,
										   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][0]],
										   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][0]],
										   words_so_far=False,
										   loading_X=True)[0][0]

				# Get (detokenized) output
				encoded_reference = target_lines[n_line]
				reference = params_prediction['detokenize_f'](encoded_reference).split() if \
					args.detokenize_bpe else encoded_reference.split()
				encoded_reference = encoded_reference.split()

				# Detokenize for nicer logging
				if args.detokenize_bpe:
					src_line = params_prediction['detokenize_f'](src_line)
				logger.debug(u'\n\nProcessing sentence %d' % (n_line + 1))
				logger.debug(u'Source: %s' % src_line)
				logger.debug(u'Target: %s' % ' '.join(reference))

				# 1. Get a first hypothesis
				trans_indices, costs, alphas = interactive_beam_searcher.sample_beam_search_interactive(src_seq)

				# 1.1. Set unk replacement strategy
				if params_prediction['pos_unk']:
					alphas    = [alphas]
					sources   = [tokenized_input]
					heuristic = params_prediction['heuristic']
				else:
					alphas    = None
					sources   = None
					heuristic = None

				# 1.2. Decode hypothesis
				encoded_hypothesis = decode_predictions_beam_search([trans_indices],
																	index2word_y,
																	alphas=alphas,
																	x_text=sources,
																	heuristic=heuristic,
																	mapping=mapping,
																	pad_sequences=True,
																	verbose=0)[0]

				# 1.3. Store results (optional)
				hypothesis = params_prediction['detokenize_f'](encoded_hypothesis) \
					if params_prediction.get('apply_detokenization', False) else encoded_hypothesis
				if args.original_dest is not None:
					filepath = args.original_dest
					if params['SAMPLING_SAVE_MODE'] == 'list':
						list2file(filepath, [hypothesis], permission='a')
					else:
						raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
				logger.debug(u'Hypo_%d: %s' % (hypothesis_number, hypothesis))
				hypothesis = hypothesis.split()
				encoded_hypothesis = encoded_hypothesis.split()

				# 2. Check Confidence Sentence Measure
				tokenized_input = tokenized_input.split()
				tokenized_input.append(CM.END_P)
				encoded_hypothesis.append(CM.END_P)

				correct_words = []
				incorrect_words = []


				# 2.2. If it is lower than the threshold edit the sentence

				# 2.2 Wrong hypothesis -> Interactively translate the sentence
				checked_index_r = 0
				checked_index_h = 0
				last_checked_index = 0
				unk_words = []
				fixed_words_user = OrderedDict()  # {pos: word}
				old_isles = []
				BPE_offset = 0
				while checked_index_r < len(reference):
					validated_prefix = []
					correction_made = False
					# Stage 1: Isles selection
					#   1. Select the multiple isles in the hypothesis.
					if not args.prefix:
						hypothesis_isles = find_isles(hypothesis, reference)[0]
						tokenized_isles = []
						next_isle_bpe_offset = 0
						for isle_idx, isle in hypothesis_isles:
							tokenized_words_in_isle = []
							for word in isle:
								tokenized_word = tokenize_f(word.encode('utf-8')).split()
								tokenized_words_in_isle += tokenized_word
							tokenized_isle = (isle_idx + next_isle_bpe_offset, tokenized_words_in_isle)
							# tokenized_isle_indices = (isle_idx + next_isle_bpe_offset, [map(lambda x: word2index_y.get(x, unk_id), tokenized_isle)])
							# logger.debug(u"tokenized_isle_indices: %s" % (str(tokenized_isle_indices)))
							next_isle_bpe_offset += len(tokenized_words_in_isle) - len(isle)
							tokenized_isles.append(tokenized_isle)
						# isle_indices =  [(index, map(lambda x: word2index_y.get(x, unk_id), word)) for index, word in hypothesis_isles]
						# hypothesis_isles_words = [(index, params_prediction['detokenize_f'](u' '.join(isle)) for index, isle in hypothesis_isles)]
						logger.debug(u"Isles: %s" % (str(hypothesis_isles)))
						isle_indices = [(index, map(lambda x: word2index_y.get(x, unk_id),
													flatten_list_of_lists(map(lambda y:
																			  tokenize_f(y).split(),
																			  word))))
										for index, word in tokenized_isles] \
							if params_prediction['apply_tokenization'] else \
							[(index, map(lambda x: word2index_y.get(x, unk_id), word))
							 for index, word in tokenized_isles]
						unks_in_isles = [(index, map(lambda w: word2index_y.get(w, unk_id), word), word) for index, word in tokenized_isles]
						# Count only for non selected isles
						# Isles of length 1 account for 1 mouse action
						mouse_actions_sentence += compute_mouse_movements(isle_indices,
																		  old_isles,
																		  last_checked_index)
					else:
						isle_indices = []
						unks_in_isles = []
					# Stage 2: INMT
					# From left to right, we will correct the hypotheses, taking into account the isles info
					# At each timestep, the user can make two operations:
					# Insertion of a new word at the end of the hypothesis
					# Substitution of a word by another
					while checked_index_r < len(reference):  # We check all words in the reference
						new_word = reference[checked_index_r]
						new_word_len = len(new_word)
						if version_info[0] < 3:  # Execute different code for python 2 or 3
							new_words = tokenize_f(new_word.encode('utf-8')).split()  # if params_prediction['apply_tokenization'] else [new_word]
						else:
							new_words = tokenize_f(str(new_word.encode('utf-8'),
													   'utf-8')).split()  # if params_prediction['apply_tokenization'] else [new_word]
						if new_words[-1][-2:] == bpe_separator:  # Remove potential subwords in user feedback.
							new_words[-1] = new_words[-1][:-2]
						
						if checked_index_h >= len(hypothesis):
							#incorrect_words.append({'word': CM.END_P,
							#						'pos':  checked_index_h + BPE_offset,
							#						'len':	len(encoded_hypothesis)})
							

							# Insertions (at the end of the sentence)
							errors_sentence += 1
							# 2.2.9 Add a mouse action if we moved the pointer
							if checked_index_h - last_checked_index > 1:
								mouse_actions_sentence += 1
							keystrokes_sentence += new_word_len
							new_word_indices = [word2index_y.get(word, unk_id) for word in new_words]
							validated_prefix.append(new_word_indices)
							for n_word, new_subword in enumerate(new_words):
								fixed_words_user[checked_index_h + BPE_offset + n_word] = new_word_indices[n_word]
								if word2index_y.get(new_subword) is None:
									if checked_index_h + BPE_offset + n_word not in unk_indices:
										unk_words.append(new_subword)
										unk_indices.append(checked_index_h + BPE_offset + n_word)
							correction_made = True
							logger.debug(u'"%s" to position %d (end-of-sentence)' % (new_word, checked_index_h))
							last_checked_index = checked_index_h

							checked_index_h += 1
							checked_index_r += 1
							BPE_offset += len(new_word_indices) - 1
							break
						elif hypothesis[checked_index_h] != reference[checked_index_r]:
							word = []
							pos = checked_index_h + BPE_offset
							for w in encoded_hypothesis[pos:]:
								if w[-2:] == bpe_separator:
									word.append(w)
								else:
									word.append(w)
									break

							incorrect_words.append({'word': word,
													'pos':  pos +1,
													'len':  len(encoded_hypothesis)})
							errors_sentence += 1
							mouse_actions_sentence += 1
							if checked_index_h - last_checked_index > 1:
								mouse_actions_sentence += 1
							last_correct_pos = checked_index_h
							keystrokes_sentence += new_word_len
							# Substitution
							new_word_indices = [word2index_y.get(word, unk_id) for word in new_words]
							validated_prefix.append(new_word_indices)
							for n_word, new_subword in enumerate(new_words):
								fixed_words_user[checked_index_h + BPE_offset + n_word] = new_word_indices[n_word]
								if word2index_y.get(new_subword) is None:
									if checked_index_h + BPE_offset + n_word not in unk_indices:
										unk_words.append(new_subword)
										unk_indices.append(checked_index_h + BPE_offset + n_word)
							correction_made = True
							logger.debug(u'"%s" to position %d' % (new_word, checked_index_h))
							last_checked_index = checked_index_h

							checked_index_h += 1
							checked_index_r += 1
							BPE_offset += len(new_word_indices) - 1
							break
						else:
							word = []
							pos = checked_index_h + BPE_offset
							for w in encoded_hypothesis[pos:]:
								if w[-2:] == bpe_separator:
									word.append(w)
								else:
									word.append(w)
									break

							correct_words.append({  'word': word,
													'pos':  pos +1,
													'len':  len(encoded_hypothesis)})
							# No errors
							if version_info[0] < 3:  # Execute different code for python 2 or 3
								correct_words_h = tokenize_f(hypothesis[checked_index_h].encode('utf-8')).split()  # if params_prediction['apply_tokenization'] else [reference[checked_index_h]]
							else:
								correct_words_h = tokenize_f(str(hypothesis[checked_index_h].encode(
									'utf-8'), 'utf-8')).split()  # if params_prediction['apply_tokenization'] else [reference[checked_index_h]]
							new_word_indices = [word2index_y.get(word, unk_id) for word in correct_words_h]
							validated_prefix.append(new_word_indices)
							for n_word, new_subword in enumerate(new_words):
								fixed_words_user[checked_index_h + BPE_offset + n_word] = new_word_indices[n_word]
								if word2index_y.get(new_subword) is None:
									if checked_index_h + BPE_offset + n_word not in unk_indices:
										unk_words.append(new_subword)
										unk_indices.append(checked_index_h + BPE_offset + n_word)

							checked_index_h += 1
							checked_index_r += 1
							BPE_offset += len(new_word_indices) - 1
							last_checked_index = checked_index_h
					old_isles = [isle[1] for isle in isle_indices]
					old_isles.append(validated_prefix)

					# Generate a new hypothesis
					if correction_made:
						logger.debug("")
						trans_indices, costs, alphas = interactive_beam_searcher. \
							sample_beam_search_interactive(src_seq,
														   fixed_words=copy.copy(fixed_words_user),
														   max_N=args.max_n,
														   isles=isle_indices,
														   idx2word=index2word_y)
						unk_in_isles = []
						for isle_idx, isle_sequence, isle_words in unks_in_isles:
							if unk_id in isle_sequence:
								unk_in_isles.append((subfinder(isle_sequence, list(trans_indices)), isle_words))
						if params['POS_UNK']:
							alphas = [alphas]
						else:
							alphas = None
						alphas = None
						encoded_hypothesis = decode_predictions_beam_search([trans_indices],
																			index2word_y,
																			alphas=alphas,
																			x_text=sources,
																			heuristic=heuristic,
																			mapping=mapping,
																			pad_sequences=True,
																			verbose=0)[0]
						encoded_hypothesis = encoded_hypothesis.split()
						hypothesis = encoded_hypothesis
						for (words_idx, starting_pos), words in unk_in_isles:
							for pos_unk_word, pos_hypothesis in enumerate(range(starting_pos, starting_pos + len(words_idx))):
								hypothesis[pos_hypothesis] = words[pos_unk_word]
						hypothesis_number += 1
						# UNK words management
						if len(unk_indices) > 0:  # If we added some UNK word
							if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
								for i, index in enumerate(range(0, len(hypothesis))):
									hypothesis[index] = unk_words[unk_indices[i]]
								for ii in range(i + 1, len(unk_words)):
									hypothesis.append(unk_words[ii])
							else:  # We put each unknown word in the corresponding gap
								for i, index in enumerate(unk_indices):
									if index < len(hypothesis):
										hypothesis[index] = unk_words[i]
									else:
										hypothesis.append(unk_words[i])
						hypothesis = u' '.join(hypothesis)
						hypothesis = params_prediction['detokenize_f'](hypothesis) \
							if params_prediction.get('apply_detokenization', False) else hypothesis
						logger.debug("Target: %s" % ' '.join(reference))
						logger.debug("Hypo_%d: %s" % (hypothesis_number, hypothesis))
						hypothesis = hypothesis.split()

				# Final check: The reference is a subset of the hypothesis: Cut the hypothesis
				if len(reference) < len(hypothesis):
					hypothesis = hypothesis[:len(reference)]
					errors_sentence += 1
					keystrokes_sentence += 1
					logger.debug("Cutting hypothesis")

				# Check confidence measures
				total_words_checked += len(correct_words) + len(incorrect_words)

				logger.debug(str([x['word'] for x in correct_words]))
				for element in correct_words:
					word = element['word']

					val = 0
					for idx, w in enumerate(word):
						word_cm = confidence_model.get_confidence(tokenized_input, w, element['pos']+idx, element['len'])
						if word_cm > val:
							val = word_cm


					for idx, threshold in enumerate(word_metrics['threshold']):
						if val >= threshold:
							word_metrics['tag_correct'][idx] += 1
						else:
							word_metrics['tag_incorrect'][idx] += 1
							word_metrics['missclasified'][idx] += 1

				logger.debug(str([x['word'] for x in incorrect_words]))
				for element in incorrect_words:
					word = element['word']

					val = 0
					for idx, w in enumerate(word):
						word_cm = confidence_model.get_confidence(tokenized_input, w, element['pos']+idx, element['len'])
						if word_cm > val:
							val = word_cm

					for idx, threshold in enumerate(word_metrics['threshold']):
						if val >= threshold:
							word_metrics['tag_correct'][idx] += 1
							word_metrics['missclasified'][idx] += 1
						else:
							word_metrics['tag_incorrect'][idx] += 1

				# 3. Update user effort counters
				total_sentences += 1
				if reference != hypothesis:
					total_wrong_sentences += 1 

				refs_metrics[n_line] = [' '.join(reference)]
				hypo_metrics[n_line] = [' '.join(hypothesis)]

				# 3.1. Calculate the scores
				scores_sentence = calculate_scores(scorers, {n_line:refs_metrics[n_line]}, {n_line:hypo_metrics[n_line]})

				# 3.2 Log some info
				logger.debug(u"Final hypotesis: %s" % u' '.join(hypothesis))

				metrics_str = ""
				for metric in scores_sentence:
					metrics_str += metric + ": " + "{:.4f}".format(scores_sentence[metric]) + "  "
				logger.debug(metrics_str + "\n\n\n\n")

				# 5 Write correct sentences into a file
				list2file(args.dest, [hypothesis], permission='a')

				if (n_line + 1) % 50 == 0:
					# PONER AQUI TAMBIEN LAS METRICAS
					logger.info(u"%d sentences processed" % (n_line + 1))
					logger.info(u"Current speed is {} per sentence".format((time.time() - start_time) / (n_line + 1)))
					logger.info(u"Current sentence CER is: %f" % (float(total_sentences - total_wrong_sentences)/float(total_sentences)))
					logger.info(u"Total number of total classified words: %d" % (total_words_checked))
					
					word_metrics['CER'] = [float(miss)/float(total_words_checked) for miss in word_metrics['missclasified']]
					for threshold, miss, correct, incorrect, cer in zip(word_metrics['threshold'], word_metrics['missclasified'], word_metrics['tag_correct'], word_metrics['tag_incorrect'], word_metrics['CER']):
						logger.info(u"Threshold: %f" % (threshold))
						logger.info(u"Total number of total wrong classified words: %d" % (miss))
						logger.info(u"Total number of total correct words: %d" % (correct))
						logger.info(u"Total number of total wrong words: %d" % (incorrect))
						logger.info(u"Current word CER is: %f" % (cer))
					export_csv('./data_{}.csv'.format(n_line), word_metrics)

					scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
					for metric in scores_sentence:
						logger.info("Current " + metric + " is: " + "{:.4f}".format(scores_sentence[metric]))

		# 6. Final!
		# 6.1 Log some information
		logger.info(u"Current sentence CER is: %f" % (float(total_sentences - total_wrong_sentences)/float(total_sentences)))
		logger.info(u"Total number of total classified words: %d" % (total_words_checked))
		word_metrics['CER'] = [float(miss)/float(total_words_checked) for miss in word_metrics['missclasified']]
		for threshold, miss, correct, incorrect, cer in zip(word_metrics['threshold'], word_metrics['missclasified'], word_metrics['tag_correct'], word_metrics['tag_incorrect'], word_metrics['CER']):
			logger.info(u"Threshold: %f" % (threshold))
			logger.info(u"Total number of total wrong classified words: %d" % (miss))
			logger.info(u"Total number of total correct words: %d" % (correct))
			logger.info(u"Total number of total wrong words: %d" % (incorrect))
			logger.info(u"Current word CER is: %f" % (cer))
		export_csv('./data.csv', word_metrics)

		scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
		for metric in scores_sentence:
						logger.info("Current " + metric + " is: " + "{:.4f}".format(scores_sentence[metric]))

		# 6.2 Close open files
		fsrc.close()
		ftrans.close()
	except KeyboardInterrupt:
		logger.debug (u'Interrupted!')
		logger.info(u"Current sentence CER is: %f" % (float(total_sentences - total_wrong_sentences)/float(total_sentences)))
		logger.info(u"Total number of total classified words: %d" % (total_words_checked))
		word_metrics['CER'] = [float(miss)/float(total_words_checked) for miss in word_metrics['missclasified']]
		for threshold, miss, correct, incorrect, cer in zip(word_metrics['threshold'], word_metrics['missclasified'], word_metrics['tag_correct'], word_metrics['tag_incorrect'], word_metrics['CER']):
			logger.info(u"Threshold: %f" % (threshold))
			logger.info(u"Total number of total wrong classified words: %d" % (miss))
			logger.info(u"Total number of total correct words: %d" % (correct))
			logger.info(u"Total number of total wrong words: %d" % (incorrect))
			logger.info(u"Current word CER is: %f" % (cer))
		export_csv('./data.csv', word_metrics)

		scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
		for metric in scores_sentence:
						logger.info("Current " + metric + " is: " + "{:.4f}".format(scores_sentence[metric]))

		# 6.2 Close open files
		fsrc.close()
		ftrans.close()

if __name__ == "__main__":
	interactive_simulation()