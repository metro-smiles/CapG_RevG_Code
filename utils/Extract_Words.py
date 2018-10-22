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


def split_words(sentnc): # Converts all words in the sentence to a list of words
	word_lst = [];
	wrd = '';

	for i in range(len(sentnc)):
		if (ord(sentnc[i]) >= 65 and ord(sentnc[i]) <= 90) or (ord(sentnc[i]) >= 97 and ord(sentnc[i]) <= 122) or ord(sentnc[i]) == '\'': # If valid character
			wrd += sentnc[i].lower()
		else:
			if wrd != '':
				word_lst.append(wrd)
			wrd = ''

	if wrd != '': # Last character of the sentence is not a special character
		word_lst.append(wrd)
		
	return word_lst
