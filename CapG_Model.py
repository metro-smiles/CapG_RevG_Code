import unicodedata
import string
import re
import random
import os
import pickle
import numpy as np
import nltk

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


#"""
#
# Define the models
class GRUModel_ContinueStop_Text(nn.Module):
	def __init__(self, hidden_size, output_size, vec_size, coher_hidden_size, topic_hidden_size, max_star, nos_imgfeat, star_op_size, cont_flag, n_layers_cont, n_layers_text, n_layers_couple):
		super(GRUModel_ContinueStop, self).__init__()
		self.n_layers_cont = n_layers_cont
		self.n_layers_text = n_layers_text
		self.hidden_size = hidden_size
		self.star_op_size = star_op_size

		#self.star_embedding = nn.Embedding(max_star, star_op_size)
		self.img_encoding = nn.Linear(nos_imgfeat, hidden_size)
		self.embedding = nn.Embedding(output_size, hidden_size) # For handling the text inputs
		self.gru_cont = nn.GRU(hidden_size, hidden_size, n_layers_cont) # GRU for start stop
		self.gru_text = nn.GRU(hidden_size, hidden_size, n_layers_text) # GRU for sentence
		self.out_cont = nn.Linear(hidden_size, cont_flag) # Flag indicating if we should continue
		self.out_text = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax()
		
		self.gru_couple = nn.GRU(vec_size, hidden_size, n_layers) # GRU for the coupling unit
		# Coherence Network
		self.fc_1_coher = nn.Linear(hidden_size, coher_hidden_size) # First Layer
		self.fc_2_coher = nn.Linear(coher_hidden_size, hidden_size) # Second Layer
		self.non_lin_coher = nn.SELU()

		# Topic Network
		self.fc_1_topic = nn.Linear(hidden_size, topic_hidden_size) # First Layer
		self.fc_2_topic = nn.Linear(topic_hidden_size, hidden_size) # Second Layer
		self.non_lin_topic = nn.SELU()

	def forward(self, input, hidden, flag):
		if flag == 'level_1': # Passing image features and stars for the first GRU of every sentence - Sentence RNN
			ip = self.img_encoding(input) # .view(1, 1, -1)
			ip = ip.view(1, 1, -1)
			#hidden = torch.cat([hidden, str_ip], 2)
			output, hidden = self.gru_cont(ip, hidden)
			output = self.softmax(self.out_cont(output[0])) # Obtain the labels of whether to continue or stop
		elif flag == 'level_1': # Passing word embeddings - Word RNN
			output = self.embedding(input).view(1, 1, -1)
			#print('Processed Input Embedding')
			output = F.relu(output)
			output, hidden = self.gru_text(output, hidden)

			output = self.softmax(self.out_text(output[0]))
		elif flag == 'couple': # Forward through the coupling unit
			output, hidden = self.gru_couple(input, hidden)
		elif flag == 'coher': # Forward through the Coherence Vector Network
			output = self.fc_1_coher(input)
			output = self.non_lin_coher(output)
			output = self.fc_2_coher(output)
			output = self.non_lin_coher(output)
			hidden = None
		elif flag == 'topic': # Forward through the Coherence Vector Network
			output = self.fc_1_topic(input)
			output = self.non_lin_topic(output)
			output = self.fc_2_topic(output)
			output = self.non_lin_topic(output)
			hidden = None
		return output, hidden
"""
class Coupling_Model(nn.Module): # GRU for the Coupling Unit
	def __init__(self, hidden_size, vec_size, max_star, nos_imgfeat, star_op_size, cont_flag, n_layers=1):
		self.gru = nn.GRU(vec_size, hidden_size, n_layers) # GRU for start stop

	def forward(self, input_vec, hidden_vec):
		output_vec, hid_vec = self.gru(input_vec, hidden_vec)
		return output_vec, hid_vec
		
class Coherence_Model(nn.Module): # NN for the Coherence Vector
	def __init__(self, hidden_size, inp_size, op_size):
		self.fc_1 = nn.Linear(inp_size, hidden_size) # First Layer
		self.fc_2 = nn.Linear(hidden_size, op_size) # Second Layer
		self.non_lin = nn.SELU()

	def forward(self, input_vec):
		output = self.fc_1(input_vec)
		output = self.non_lin(output)
		output = self.fc_2(output)
		output = self.non_lin(output)
		return output

class Topic_Net(nn.Module): # NN for the Topic Vector Generation
	def __init__(self, hidden_size, inp_size, op_size):
		self.fc_1 = nn.Linear(inp_size, hidden_size) # First Layer
		self.fc_2 = nn.Linear(hidden_size, op_size) # Second Layer
		self.non_lin = nn.SELU()

	def forward(self, input_vec):
		output = self.fc_1(input_vec)
		output = self.non_lin(output)
		output = self.fc_2(output)
		output = self.non_lin(output)
		return output
"""