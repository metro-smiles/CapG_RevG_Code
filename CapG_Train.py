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
from torch.autograd import 

from torch import optim
import torch.nn.functional as F

#base_dir = '/path/to/folder/containing/training/and/test/data/'

# Define the actual training function -- operates one training sample pair (x, y) at a time
def train(input_variable, target_variable, obj_name, model, criterion_1, criterion_2, hidden_size, opt, base_dir, loss = 0):
	model_hidden_st = None # Stores the hidden state vector at every step of the Sentence RNN
	
	nos_sentc = len(input_variable); sent_exec = 0; 
	
	sent_cand = 0;
	for st in range(opt.MAX_SENTC): # Iterate to see how many sentences the model intends to generate

		# Read the Image Features
		if os.path.isfile(base_dir + 'Train/' +  obj_name + '/Feat_Vec.pickle') == False: # Check if image feature file is present
			return

		with open(base_dir + 'Train/' +  obj_name + '/Feat_Vec.pickle') as f: # Read in the review
			feats = pickle.load(f)

		mod_feats = np.zeros((1, feats.shape[1] ), dtype = np.float32) # + 1 

		# Copy the features
		for i in range(feats.shape[1]):
			mod_feats[0, i] = feats[0, i]

		temp_ip = torch.from_numpy(mod_feats)
		temp_ip = temp_ip.float()
		mod_ip = Variable(temp_ip, requires_grad=True) # Push in the Image Feature Here

		if st == 0:
			temp_hid = np.zeros(hidden_size, dtype = np.float32) # random.uniform(0, 1, (hidden_size - star_embed ) )
			temp_hid = temp_hid.reshape(1, 1, hidden_size )
			model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True)
		else:
			mh = model_hidden_st.cpu().data.numpy()
			model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size].reshape(1, 1, hidden_size) ), requires_grad=True)

		# Check if Variable should be moved to GPU
		if opt.USE_CUDA:
			mod_ip = mod_ip.cuda()
			model_hidden = model_hidden.cuda()

		# Call the model for the first time at the beginning of a sentence
		output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # Indicating that the first level RNN is to be used
		model_hidden_st = model_hidden
		strtstp_topv, strtstp_topi = output_contstop.data.topk(1)
		strtstp_ni = strtstp_topi[0][0]

		if strtstp_ni == 0: # So we continue
			sent_cand += 1
	
	loss += opt.L_S * criterion_1(sent_cand, nos_sentc) # The cross-entropy loss over the number of sentences
	
	# Count the number of valid sentences in the training sample
	val_sent = 0;
	for st in range(nos_sentc): # Count the number of valid sentence
		if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
			continue
		if os.path.isfile(base_dir + 'Train/' + obj_name + '/Feat_Vec.pickle') == False: # Check if image feature file is present
			return
		val_sent += 1

	# Create the array of topic vectors and construct the Global Topic Vector - Topic Generation Net
	gl_mh = np.zeros((1, 1, hidden_size, val_sent))
	model_hidden_st = None
	# Stack up the vectors
	for st in range(nos_sentc): # Iterate over each sentence separately
		if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
			continue
		# Read the Image Features
		if os.path.isfile(base_dir + 'Train/' + obj_name + '/Feat_Vec.pickle') == False: # Check if image feature file is present
			return

		with open(base_dir + 'Train/' +  obj_name + '/Feat_Vec.pickle') as f: # Read in the Image Features
			feats = pickle.load(f)

		mod_feats = np.zeros((1, feats.shape[1] ), dtype = np.float32)

		# Copy the features
		for i in range(feats.shape[1]):
			mod_feats[0, i] = feats[0, i]

		temp_ip = torch.from_numpy(mod_feats)
		temp_ip = temp_ip.float()
		mod_ip = Variable(temp_ip, requires_grad=True)

		if sent_exec == 0: # The first sentence
			temp_hid = np.zeros(hidden_size, dtype = np.float32) # random.uniform(0, 1, (hidden_size) )
			temp_hid = temp_hid.reshape(1, 1, hidden_size )
			model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True) # Push in the Image Feature Here #encoder_hidden
			sent_exec += 1
		else: # All other sentences are initialized from previous sentences
			mh = model_hidden_st.cpu().data.numpy()
			model_hidden =  Variable(torch.from_numpy( mh[0, 0, :hidden_size].reshape(1, 1, hidden_size) ), requires_grad=True) # Obtain the hidden state from the previous hidden state
			sent_exec += 1

		# Check if Variable should be moved to GPU
		if opt.USE_CUDA:
			mod_ip = mod_ip.cuda()
			model_hidden = model_hidden.cuda()

		output_contstop, model_hidden = model(mod_ip, model_hidden, 'level_1') # level_1 indicates that we are using the Senetence RNN
		model_hidden_st = model_hidden
		gl_mh[0, 0, :, sent_exec-1] = (model(model_hidden_st, None, 'topic')[0].cpu().data.numpy()).reshape(1, 1, hidden_size) # Transform the hidden state to obtain the topic vector
	
	# Compute the global topic vector as a weighted average of the individual topic vectors
	glob_vec = gl_mh[0, 0, :, 0].reshape(1, 1, hidden_size)
	for i in range(1, val_sent):
		glob_vec[:, :, :] += gl_mh[:, :, :, i].reshape(1, 1, hidden_size) * (LA.norm(gl_mh[:, :, :, i].reshape(-1)) / np.sum(LA.norm(gl_mh[:, :, :, :].reshape(-1, val_sent).T, axis=1)))


	# Process the Sentence RNN
	#Previous Hidden State Vector - The Coherence Vector
	prev_vec = ( np.zeros((1, 1, hidden_size)) ).astype(np.float32)

	for st in range(nos_sentc): # Iterate over each sentence separately
		
		if len(input_variable[st]) <= 1: # If the sentence is of unit length, skip it
			continue
		ip_var = Variable(torch.LongTensor( input_variable[st] ), requires_grad=True) # One sentence
		op_var = Variable(torch.LongTensor( target_variable[st] ), requires_grad=True)
		input_length = ip_var.size()[0]
		target_length = op_var.size()[0]
		
		loc_vec = (gl_mh[:, :, :, st]).reshape(1, 1, -1) # The original topic vector for the current sentence
		comb = np.add((1 - opt.lamb) * loc_vec[0, 0, :], (opt.lamb) * prev_vec[0, 0, :]) # Combine the current topic vector and the coherence vector from the previous sentence
		mh = ((model(glob_vec[0, 0, :], comb, 'couple' )[0] ).reshape(1, 1, -1)).astype(np.float32) # Coupling Unit
		mh = (( comb  ).reshape(1, 1, -1)).astype(np.float32) 

		# Construct the input for the first word of a sentence in the Sentence RNN
		model_input =  Variable(torch.from_numpy( mh[0, 0, :].reshape(1, 1, mod_feats.shape[1]) ), requires_grad=True)
		model_hidden = Variable(torch.from_numpy(temp_hid), requires_grad=True)

		if opt.USE_CUDA:
			model_hidden = model_hidden.cuda()
			model_input = model_input.cuda()
			ip_var = ip_var.cuda()
			op_var = op_var.cuda()
			
		# Teacher forcing: Feed the target as the next input
		for di in xrange(1, target_length, 1):
			#print('Word Number: ' + str(di))
			model_output, model_hidden = model(model_input, model_hidden, 'level_2') # level_2 indicates that we want to use the Sentence RNN
			loss += opt.L_W * criterion_2(model_output, op_var[di:di+1]) # Use the second cross-entropy term
			model_input = op_var[di] # Teacher forcing
		
		# Re-initialize the previous vector
		prev_vec = model(model_hidden, None, 'coher')[0]
		
	return loss.data[0] * 1.0 / target_length, loss # Pass the loss information to the calling function
