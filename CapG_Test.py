import unicodedata
import string
import re
import random
import os
import pickle
import numpy as np
import nltk
from numpy import linalg as LA

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# base_dir = '/path/to/folder/containing/training/and/test/data/'

# The function for testing
def test(obj_name, model, opt, base_dir):
	
	model_hidden_st = None # Stores the hidden state vector at every step of the Sentence RNN
	pred_words = [] # Stores the list of synthesized words

	# Create the array of topic vectors and the Global Topic Vector -- Topic Generation Net
	gl_mh = np.zeros((1, 1, opt.hidden_size, MAX_SENTC)); val_sent = 0;

	for st in range(opt.MAX_SENTC): # Iterate over each sentence separately

		# Read the Image Features
		if os.path.isfile(base_dir + 'Test/' +  obj_name + '/Feat_Vec.pickle') == False: # Check if image feature file is present
			return

		with open(base_dir + 'Test/' +  obj_name + '/Feat_Vec.pickle') as f: # Read in the feature
			feats = pickle.load(f)

		mod_feats = np.zeros((1, feats.shape[1] ), dtype = np.float32) # + 1 

		# Copy the features
		for i in range(feats.shape[1]):
			mod_feats[0, i] = feats[0, i]

		temp_ip = torch.from_numpy(mod_feats)
		temp_ip = temp_ip.float()
		mod_ip = Variable(temp_ip) # Push in the Image Feature Here

		if st == 0: # Initialize the hidden state for the first se
			temp_hid = np.zeros(opt.hidden_size, dtype = np.float32) # random.uniform(0, 1, (opt.hidden_size - star_embed ) )
			temp_hid = temp_hid.reshape(1, 1, opt.hidden_size )
			model_hidden = Variable(torch.from_numpy(temp_hid))
		else:
			mh = model_hidden_st.cpu().data.numpy()
			model_hidden =  Variable(torch.from_numpy( mh[0, 0, :opt.hidden_size].reshape(1, 1, opt.hidden_size) ))

		# Check if Variable should be moved to GPU
		if opt.USE_CUDA:
			mod_ip = mod_ip.cuda()
			model_hidden = model_hidden.cuda()
		
		output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # level_1 indicates that we are using the Senetence RNN
		model_hidden_st = model_hidden
		strtstp_topv, strtstp_topi = output_contstop.data.topk(1)
		strtstp_ni = strtstp_topi[0][0]

		if strtstp_ni == 0: # So we continue
			val_sent += 1
			gl_mh[0, 0, :, st] = (model(model_hidden_st, None, 'topic')[0].cpu().data.numpy()).reshape(1, 1, opt.hidden_size) # Transform the hidden state to obtain the topic vector

	# Compute the Global Topic Vector as a weighted average of the individual topic vectors
	glob_vec = gl_mh[0, 0, :, 0].reshape(1, 1, opt.hidden_size)
	for i in range(1, val_sent):
		glob_vec[:, :, :] += glob_vec[:, :, :] += gl_mh[:, :, :, i].reshape(1, 1, opt.hidden_size) * (LA.norm(gl_mh[:, :, :, i].reshape(-1)) / np.sum(LA.norm(gl_mh[:, :, :, :].reshape(-1, val_sent).T, axis=1)))

	# Sentence Generation Net
	#Previous Hidden State Vector
	prev_vec = ( np.zeros((1, 1, opt.hidden_size)) ).astype(np.float32)

	for st in range(opt.MAX_SENTC): # Iterate over each sentence separately and generate the words

		loc_vec = (gl_mh[:, :, :, st]).reshape(1, 1, -1) # The original topic vector for the current sentence
		comb = np.add((1-opt.lamb) * loc_vec[0, 0, :], (opt.lamb) * prev_vec[0, 0, :]) # Combine the current topic vector and the coherence vector from the previous sentence
		mh = ((model(glob_vec[0, 0, :], comb, 'couple')[0] ).reshape(1, 1, -1)).astype(np.float32) # Coupling Unit
		mh = (( comb  ).reshape(1, 1, -1)).astype(np.float32) 

		# Construct the input for the first word of a sentence in the Sentence RNN
		model_input =  Variable(torch.from_numpy( mh[0, 0, :].reshape(1, 1, mod_feats.shape[1]) ))
		model_hidden = Variable(torch.from_numpy(temp_hid))

		if opt.USE_CUDA:
			model_hidden = model_hidden.cuda()
			model_input = model_input.cuda()
		
		for di in range(opt.max_length):
			
			model_output, model_hidden = model(model_input, model_hidden, 'level_2') # level_2 indicates that we want to use the Sentence RNN
			topv, topi = model_output.data.topk(1) # Standard RNN decoding of the words
			ni = topi[0][0]

			# Check if EOS has been predicted
			if ni == len(wrd_list):
				pred_words.append('<EOS>')
				break
			else:
				pred_words.append(wrd_list[ni])
				model_input = Variable(torch.LongTensor( [ni] ))

		# Re-initialize the previous vector
		prev_vec = model(model_hidden, None, 'coher')[0]
	return pred_words