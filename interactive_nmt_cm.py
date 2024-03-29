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
from utils.confidence_measure import CM, IBM1, IBM2, Fast_Align, HMM
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
	parser.add_argument("-ma", 								type=int, 	required=False,	default=0, 		help="Max number of mouse actions for the same position")

	parser.add_argument("-est", "--confidence_estimator", 	type=int, 	required=False, default=0, 		help="Confidence Estimator to use. 0-IBM1 | 1-IBM2 | 2-Fast_Align | 3-HMM")
	parser.add_argument("-cm_output",						type=str, 	required=True,					help="Ouptut File for the data")
	parser.add_argument("-cm_alpha", 						type=float,	required=False,	default=0.5,	help="Lineal combination alpha")
	parser.add_argument("-cm_sentence", 					action='store_true', 		default=False, 	help="Ratio Confidence Measure")
	parser.add_argument("-lm", "--lexicon_model", 			type=str,	required=True, 					help="Path to the model of the confidence measure")
	parser.add_argument("-am", "--alignment_model",  		type=str, 	required=True, 					help="Path to the alignment model of the confidence measure")
	parser.add_argument("-rt", "--ratio_threshold", 		type=float, required=False,	default=0.0, 	help="Threshold for ratio mode")
	parser.add_argument("-st", "--sentence_threshold", 		type=float, required=False,	default=0.0, 	help="Sentence threshold")
	parser.add_argument("-wt", "--word_threshold", 			type=float, required=False,	default=1.0, 	help="Words threshold")

	return parser.parse_args()

def get_sentence_cm(confidence_model, source, hypothesis, ratio_method=False, ratio=0.0):
	sentence_cm = 0.0

	if not ratio_method:
		sentence_cm = confidence_model.get_mean_confidence(source, hypothesis)
	elif ratio_method:
		sentence_cm = confidence_model.get_ratio_confidence(source, hypothesis, ratio)

	return sentence_cm

def get_word_cm(confidence_model, source, hypo_word, position, length):
	word_cm = confidence_model.get_confidence(source, hypo_word, position, length)
	return word_cm

def export_csv(output_path, data):
    # threshold, missclassified, tag_correct, tag_incorrect, CER
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


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
	mode = args.confidence_estimator
	if   mode == 0:
		confidence_model = IBM1(args.lexicon_model)
	elif mode == 1:
		confidence_model = IBM2(args.lexicon_model, args.alignment_model, alpha=args.cm_alpha)
	elif mode == 2:
		confidence_model = Fast_Align(args.lexicon_model, alpha=args.cm_alpha)
	elif mode == 3:
		confidence_model = HMM(args.lexicon_model, args.alignment_model, alpha=args.cm_alpha)
	else:
		exit()
	sentence_threshold = args.sentence_threshold
	word_threshold = args.word_threshold

	# Initialize counters
	total_errors = 0
	total_words = 0
	total_chars = 0
	total_mouse_actions = 0
	total_keystrokes = 0
	total_sentences = 0
	total_wrong_sentences = 0
	total_words_checked = 0
	total_wrong_words = 0

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

			data = {
				 'errors': 		[],
				 'selections': 	[],
				 'WSR': 		[],
				 'MAR': 		[],
				 'uMAR': 		[],
				 'MAR_c': 		[],
				 'sCER': 		[],
				 'wCER': 		[],
				 'Bleu_1':		[],
				 'Bleu_2':		[],
				 'Bleu_3':		[],
				 'Bleu_4':		[]
				}

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

				# 2. Check Confidence Sentence Measure
				#encoded_hypothesis.append(CM.END_P)
				src_line = src_line.split()
				src_line.append(CM.END_P)
				sentence_cm = get_sentence_cm(confidence_model, src_line, hypothesis, args.cm_sentence, args.ratio_threshold)

				logger.debug(u"Sentence confidence Measure: %.10f" % sentence_cm) 
				if sentence_cm >= sentence_threshold:
					# 2.1. If it is greater or equal than the threshold check it as correct
					pass
				else:
					# 2.2. If it is lower than the threshold edit the sentence

					#2.3. Initialice edit variables
					checked_index_r = 0
					unk_words = []
					fixed_words_user = OrderedDict() # {pos: word}
					BPE_offset = 0

					#2.4. Check every word of the hypothesis
					while checked_index_r <= len(hypothesis):
						correction_made = False
						while checked_index_r <= len(hypothesis):

							# 2.5 Check Confidence Word Measure
							correct_word = True
							if checked_index_r == len(hypothesis):
								hypo_word = ''
								hypo_words = []

								word_cm = get_word_cm(confidence_model, src_line, CM.END_P, checked_index_r, len(hypothesis))
								if word_cm < word_threshold:
									correct_word = False
							else:
								# Tokenize if necessary the hypothesis word
								hypo_word = hypothesis[checked_index_r]
								if version_info[0] < 3:  # Execute different code for python 2 or 3
									hypo_words = tokenize_f(hypo_word.encode('utf-8')).split() 
								else:
									hypo_words = tokenize_f(str(hypo_word.encode('utf-8'), 'utf-8')).split()

								word_cm = get_word_cm(confidence_model, src_line, hypo_word, checked_index_r, len(hypothesis))

								correct_word = True
								if word_cm < word_threshold:
									correct_word = False

							if not correct_word:
								# 2.6. The measure confidence of the word is lower than the threshold

								if checked_index_r >= len(reference):
									# The hypothesis is to long

									hypothesis = hypothesis[:checked_index_r]
									errors_sentence += 1
									keystrokes_sentence += 1
									mouse_actions_sentence += 1
									logger.debug("Cutting hypothesis")

									total_words_checked += 1
									if hypo_word == '':
										total_wrong_words += 1

									checked_index_r += 1
									break

								elif hypo_word != reference[checked_index_r]:
									# The hypothesis word is incorrect
										correction_made = True

										# Tokenize the correct word from the reference
										ref_word = reference[checked_index_r]
										if version_info[0] < 3:  # Execute different code for python 2 or 3
											ref_words = tokenize_f(ref_word.encode('utf-8')).split() 
										else:
											ref_words = tokenize_f(str(ref_word.encode('utf-8'), 'utf-8')).split()

										# Add the word tot he fixed_words_user
										new_word_indices = [word2index_y.get(word, unk_id) for word in ref_words]
										for n_word, new_subword in enumerate(ref_words):
											fixed_words_user[checked_index_r + BPE_offset + n_word] = new_word_indices[n_word]
											if word2index_y.get(new_subword) is None:
												if checked_index_r + BPE_offset + n_word not in unk_indices:
													unk_words.append(new_subword)
													unk_indices.append(checked_index_r + BPE_offset + n_word)

										# Update the interactive variables
										errors_sentence += 1
										mouse_actions_sentence += 1
										keystrokes_sentence += len(ref_word)
										total_words_checked += 1

										# Change to the next word
										checked_index_r += 1
										BPE_offset += len(ref_words) -1


										# Show some info
										logger.debug(u'"%s" to position %d: %.10f Confidence' % (ref_word, checked_index_r, word_cm))

										break
								else:
									# The hypothesis word is correct

									# Add the word tot he fixed_words_user
									new_word_indices = [word2index_y.get(word, unk_id) for word in hypo_words]
									for n_word, new_subword in enumerate(hypo_words):
										fixed_words_user[checked_index_r + BPE_offset + n_word] = new_word_indices[n_word]
										if word2index_y.get(new_subword) is None:
											if checked_index_r + BPE_offset + n_word not in unk_indices:
												unk_words.append(new_subword)
												unk_indices.append(checked_index_r + BPE_offset + n_word)

									# Update the interactive variables
									total_words_checked += 1
									total_wrong_words += 1

									# Change to the next word
									checked_index_r += 1
									BPE_offset += len(hypo_words) -1
							else:
								# 2.7. The measure confidence of the word is greater than the threshold

								# Add the word tot he fixed_words_user
								new_word_indices = [word2index_y.get(word, unk_id) for word in hypo_words]
								for n_word, new_subword in enumerate(hypo_words):
									fixed_words_user[checked_index_r + BPE_offset + n_word] = new_word_indices[n_word]
									if word2index_y.get(new_subword) is None:
										if checked_index_r + BPE_offset + n_word not in unk_indices:
											unk_words.append(new_subword)
											unk_indices.append(checked_index_r + BPE_offset + n_word)

								# Update the interactive variables
								total_words_checked += 1
								if checked_index_r >= len(reference) or hypo_word != reference[checked_index_r]:
									total_wrong_words += 1

								# Change to the next word
								checked_index_r += 1
								BPE_offset += len(hypo_words) -1
							
						# Generate a new hypothesis
						if correction_made:
							logger.debug("")
							trans_indices, costs, alphas = interactive_beam_searcher. \
								sample_beam_search_interactive(src_seq,
																fixed_words=copy.copy(fixed_words_user),
																max_N=args.max_n,
																#isles=isle_indices,
																idx2word=index2word_y,
																#excluded_words=wrong_words
																)

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
							encoded_hypothesis.append(CM.END_P)


				# 3. Update user effort counters
				mouse_actions_sentence += 1  # This +1 is the validation action
				chars_sentence = len(u' '.join(reference))
				words_sentence = len(reference)

				total_words += len(reference)
				total_errors += errors_sentence
				total_chars += chars_sentence
				total_keystrokes += keystrokes_sentence
				total_mouse_actions += mouse_actions_sentence

				total_sentences += 1
				if reference != hypothesis:
					total_wrong_sentences += 1 

				refs_metrics[n_line] = [' '.join(reference)]
				hypo_metrics[n_line] = [' '.join(hypothesis)]

				# 3.1. Calculate the scores
				scores_sentence = calculate_scores(scorers, {n_line:refs_metrics[n_line]}, {n_line:hypo_metrics[n_line]})

				# 3.2 Log some info
				logger.debug(u"Final hypotesis: %s" % u' '.join(hypothesis))
				logger.debug(u"%d errors. "
							 u"Sentence WSR: %4f. "
							 u"Sentence mouse strokes: %d "
							 u"Sentence MAR: %4f. "
							 u"Sentence uMar: %4f. "
							 u"Sentence MAR_c: %4f. "
							 u"Sentence **KSMR**: %4f. "
							 u"Accumulated (should only be considered for debugging purposes!) WSR: %4f. "
							 u"MAR: %4f. "
							 u"MAR_c: %4f."
							 u"**KSMR**: %4f." %
							 (errors_sentence,
							  float(errors_sentence) / words_sentence,
							  mouse_actions_sentence,
							  float(mouse_actions_sentence) / words_sentence,
							  (float(total_mouse_actions) - args.ma*float(errors_sentence))/float(total_mouse_actions),
							  float(mouse_actions_sentence) / chars_sentence,
							  float(keystrokes_sentence + mouse_actions_sentence) / chars_sentence,
							  float(total_errors) / total_words,
							  float(total_mouse_actions) / total_words,
							  float(total_mouse_actions) / total_chars,
							  float(total_keystrokes + total_mouse_actions) / total_chars,
							  ))


				metrics_str = ""
				for metric in scores_sentence:
					metrics_str += metric + ": " + "{:.4f}".format(scores_sentence[metric]) + "  "
				logger.debug(metrics_str + "\n\n\n\n")

				#if scores_sentence["Bleu_1"] < 0.9990:
				#	exit()

				# 4. If we are performing OL after each correct sample:
				if args.online:
					# 4.1 Compute model inputs
					# 4.1.1 Source text -> Already computed (used for the INMT process)
					# 4.1.2 State below
					state_below = dataset.loadText([encoded_reference],
												   vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
												   max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
												   offset=1,
												   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][-1]],
												   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][-1]],
												   words_so_far=False,
												   loading_X=True)[0]

					# 4.1.3 Ground truth sample -> Interactively translated sentence
					trg_seq = dataset.loadTextOneHot([encoded_reference],
													 vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
													 vocabulary_len=dataset.vocabulary_len[
														 params['OUTPUTS_IDS_DATASET'][0]],
													 max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
													 offset=0,
													 fill=dataset.fill_text[params['OUTPUTS_IDS_DATASET'][0]],
													 pad_on_batch=dataset.pad_on_batch[params['OUTPUTS_IDS_DATASET'][0]],
													 words_so_far=False,
													 sample_weights=params['SAMPLE_WEIGHTS'],
													 loading_X=False)
					# 4.2 Train online!
					online_trainer.train_online([np.asarray([src_seq]), state_below], trg_seq,
												trg_words=[reference])
				# 5 Write correct sentences into a file
				list2file(args.dest, [hypothesis], permission='a')

				if (n_line + 1) % 50 == 0:
					# PONER AQUI TAMBIEN LAS METRICAS
					scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
					data['errors'].append(total_errors)
					data['selections'].append(total_mouse_actions)
					data['WSR'].append(float(total_errors)/total_words)
					data['MAR'].append(float(total_mouse_actions)/total_words)
					data['uMAR'].append((float(total_mouse_actions) - args.ma*float(total_errors))/float(total_mouse_actions))
					data['MAR_c'].append(float(total_mouse_actions) / total_chars)
					data['sCER'].append(float(total_sentences - total_wrong_sentences)/float(max(total_sentences,1)))
					data['wCER'].append(float(total_words_checked - total_wrong_words)/float(max(total_words_checked,1)))
					for metric in scores_sentence:
						data[metric].append(scores_sentence[metric])

					# 6.1 Log some information
					logger.info(u"%d sentences processed" % (n_line + 1))
					logger.info(u"Current speed is {} per sentence".format((time.time() - start_time) / (n_line + 1)))
					logger.debug (u"Total number of errors: %d" % (data['errors'][-1]))
					logger.debug (u"Total number selections: %d" % (data['selections'][-1]))
					logger.debug (u"WSR: %f" % (data['WSR'][-1]))
					logger.debug (u"MAR: %f" % (data['MAR'][-1]))
					logger.info(u"Current uMAR is: %f" % (data['uMAR'][-1]))
					logger.debug (u"MAR_c: %f" % (data['MAR_c'][-1]))
					logger.debug (u"**KSMR**: %f" % (float(total_keystrokes + total_mouse_actions) / total_chars))
					logger.info(u"Current sentence CER is: %f" % (data['sCER'][-1]))
					logger.info(u"Current word CER is: %f" % (data['wCER'][-1]))
					for metric in scores_sentence:
						logger.info("Current " + metric + " is: " + "{:.4f}".format(data[metric][-1]))

		# 6. Final!
		scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
		data['errors'].append(total_errors)
		data['selections'].append(total_mouse_actions)
		data['WSR'].append(float(total_errors)/total_words)
		data['MAR'].append(float(total_mouse_actions)/total_words)
		data['uMAR'].append((float(total_mouse_actions) - args.ma*float(total_errors))/float(total_mouse_actions))
		data['MAR_c'].append(float(total_mouse_actions) / total_chars)
		data['sCER'].append(float(total_sentences - total_wrong_sentences)/float(max(total_sentences,1)))
		data['wCER'].append(float(total_words_checked - total_wrong_words)/float(max(total_words_checked,1)))
		for metric in scores_sentence:
			data[metric].append(scores_sentence[metric])

		# 6.1 Log some information
		logger.debug (u"Total number of errors: %d" % (data['errors'][-1]))
		logger.debug (u"Total number selections: %d" % (data['selections'][-1]))
		logger.debug (u"WSR: %f" % (data['WSR'][-1]))
		logger.debug (u"MAR: %f" % (data['MAR'][-1]))
		logger.info(u"Current uMAR is: %f" % (data['uMAR'][-1]))
		logger.debug (u"MAR_c: %f" % (data['MAR_c'][-1]))
		logger.debug (u"**KSMR**: %f" % (float(total_keystrokes + total_mouse_actions) / total_chars))
		logger.info(u"Current sentence CER is: %f" % (data['sCER'][-1]))
		logger.info(u"Current word CER is: %f" % (data['wCER'][-1]))
		for metric in scores_sentence:
			logger.info("Current " + metric + " is: " + "{:.4f}".format(data[metric][-1]))

		export_csv(args.cm_output, data)

		# 6.2 Close open files
		fsrc.close()
		ftrans.close()
	except KeyboardInterrupt:
		scores_sentence = calculate_scores(scorers, refs_metrics, hypo_metrics)
		data['errors'].append(total_errors)
		data['selections'].append(total_mouse_actions)
		data['WSR'].append(float(total_errors)/total_words)
		data['MAR'].append(float(total_mouse_actions)/total_words)
		data['uMAR'].append((float(total_mouse_actions) - args.ma*float(total_errors))/float(total_mouse_actions))
		data['MAR_c'].append(float(total_mouse_actions) / total_chars)
		data['sCER'].append(float(total_sentences - total_wrong_sentences)/float(max(total_sentences,1)))
		data['wCER'].append(float(total_words_checked - total_wrong_words)/float(max(total_words_checked,1)))
		for metric in scores_sentence:
			data[metric].append(scores_sentence[metric])

		# 6.1 Log some information
		logger.debug (u'Interrupted!')
		logger.debug (u"Total number of errors: %d" % (data['errors'][-1]))
		logger.debug (u"Total number selections: %d" % (data['selections'][-1]))
		logger.debug (u"WSR: %f" % (data['WSR'][-1]))
		logger.debug (u"MAR: %f" % (data['MAR'][-1]))
		logger.info(u"Current uMAR is: %f" % (data['uMAR'][-1]))
		logger.debug (u"MAR_c: %f" % (data['MAR_c'][-1]))
		logger.debug (u"**KSMR**: %f" % (float(total_keystrokes + total_mouse_actions) / total_chars))
		logger.info(u"Current sentence CER is: %f" % (data['sCER'][-1]))
		logger.info(u"Current word CER is: %f" % (data['wCER'][-1]))
		for metric in scores_sentence:
			logger.info("Current " + metric + " is: " + "{:.4f}".format(data[metric][-1]))

		export_csv(args.cm_output, data)

		# 6.2 Close open files
		fsrc.close()
		ftrans.close()

if __name__ == "__main__":
	interactive_simulation()