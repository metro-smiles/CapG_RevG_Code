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


def rotate(l, n): # Rotate a list
    return l[n:] + l[:n]
#"""
# Load the Dictionary:
with open( base_dir + './Word_Dict.pickle') as f: # Load the Word List #base_dir +
	wrd_list = pickle.load(f)

def build_lists(base_dir, wrd_lst):
	glob_rev_list = []; 
	# Represent the training set as indices of dictionary elements
	for dirs in list_fldr: #[:train_samp]: # Operate only on the training set
		print('In :' + str(dirs))
		directory = base_dir + 'Train/' + str(dirs)
		file_list = os.listdir(directory);
		
		
		nos_rev = 5; cnt = 0;

		for files in file_list: # For each object pick only nos_rev reviews # [:nos_rev]

			if files.startswith('Rev'): # If it is a caption or a review read it
				cnt += 1
				with open(directory + '/' + files) as f: # Read in the review
					rev = pickle.load(f)
			else: # If the current file is not a review
				continue

			#print(rev)
			# Process the review
			wrd = ''; sntc = 0; rev_enc = {}; t_rev_enc = {}; train_rev_enc = []; #wrd_enc = np.zeros((len(wrd_list))); 
			rev_enc[sntc] = []; t_rev_enc[sntc] = [];
			for i in range(len(rev)): 
				if (( ord(rev[i]) >= 65 ) and ( ord(rev[i]) <= 90 )) or (( ord(rev[i]) >= 97 ) and ( ord(rev[i]) <= 122 )) or (ord(rev[i]) == 39): # Check for the apostrophe
					wrd += str(rev[i].lower())
				elif wrd != '': # There is a current word
					if wrd in wrd_list: # Check if the current word is in the list
						rev_enc[sntc].append(wrd_list.index(wrd)) #rev_enc.append(wrd_enc) # Add it to the list of one-hot encoded vectors

					wrd = '';
				if ( rev[i] == '.' ):
					sntc += 1;
					if (i != len(rev) - 1): # It is not the last stop
						rev_enc[sntc] = []; t_rev_enc[sntc] = [];

			if wrd != '': # The final word was not written
				if wrd in wrd_list: # Check if the current word is in the list
					rev_enc[sntc].append(wrd_list.index(wrd)) #rev_enc.append(wrd_enc) # Add it to the list of one-hot encoded vectors

				wrd = ''
			#print('Last Word in the List: ' + wrd_list[len(wrd_list) - 1]) # Last Word After Every Review

			for st in range(sntc): # For each sentence
				# Append the end symbol # Save the One-Hot Encoding:
				rev_enc[st].append(len(wrd_list)) # For target append the Start/End Symbol at the end
				t_rev_enc[st] = rotate(rev_enc[st], -1) # For input append the Start/End Symbol at the beginning

			# Construct triplets of (input seq., target seq., and object name)
			train_rev_enc.append(t_rev_enc) # Input List is appended first
			train_rev_enc.append(rev_enc) # Next we append the target
			#train_rev_enc.append(star) # Include the number of stars in the review
			train_rev_enc.append(str(dirs)) # Finally the name of the object			
			glob_rev_list.append(train_rev_enc) # Append everything to the full list of reviews (i.e. the training set)
