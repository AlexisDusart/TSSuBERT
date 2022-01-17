# Libraries import

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout, Input, Concatenate
from tensorflow.keras import regularizers
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
import numpy as np
from tqdm import tqdm
import pickle5 as pickle
import json
import time
from dateutil import tz
from datetime import datetime
import unidecode
import html
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Retrieve the existing tokenizer and pre-trained model from DistilBert.

dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Function for model creation

def create_model():
	inps = Input(shape = (max_len,), dtype='int64')
	masks= Input(shape = (max_len,), dtype='int64')
	dbert_layer = dbert_model(inps, attention_mask=masks)[0][:,0,:]
	freq_layer = Input(shape = (len(dbert_tokenizer.vocab)-1), dtype='float64') # the first token (PAD token) is not used in the tokens frequencies
	dense0 = Dense(50,activation='relu',kernel_regularizer=regularizers.l2(0.01))(freq_layer)
	dropout0= Dropout(0.5)(dense0)
	concatted = Concatenate()([dbert_layer, dropout0])
	dense = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(concatted)
	dropout= Dropout(0.5)(dense)
	pred = Dense(1, activation='relu',kernel_regularizer=regularizers.l2(0.01))(dropout)
	model = tf.keras.Model(inputs=[inps,masks,freq_layer], outputs=pred)
	print(model.summary())
	return model

# Tokenizer vocabulary update (add htg, mtn, url and rtw tokens)

dbert_tokenizer.vocab["htg"]=len(dbert_tokenizer.vocab)
dbert_tokenizer.vocab["mtn"]=len(dbert_tokenizer.vocab)
dbert_tokenizer.vocab["url"]=len(dbert_tokenizer.vocab)
dbert_tokenizer.vocab["rtw"]=len(dbert_tokenizer.vocab)

dbert_model.resize_token_embeddings(len(dbert_tokenizer))

max_len = 50 # Max length set to 50 to take advantage of the tweets length
model=create_model()


# Endode tweets into bert tokens:
# Example of code to create list of ids and attention masks
# First we define classes for preprocessing and tweet objects
# Second we load tweets (json files) with preprocessing
# Then we create list of ids and attention masks

class Preprocessing(object):
	"""
	Functions for tweets preprocessing.
	"""
	__slots__ = ("expansions","substitutions","tweets_tokens","punctuations","normalizes")

	def __init__(self,expansions=True,substitutions=True,tweets_tokens=True,punctuations=True,normalizes=True):
		"""
		Choose preprocess you want: expansions (e.g. amaaaaaaaaaaazing), substitutions (e.g. b4), tweets specifics/features tokens (@,#,RT,URL).
		"""
		self.expansions = expansions
		self.substitutions = substitutions
		self.tweets_tokens = tweets_tokens
		self.punctuations = punctuations
		self.normalizes = normalizes

	def normalize(self,tweet):
		"""
		Normalize text to ASCII representation.
		"""
		tweet.text = unidecode.unidecode(tweet.text)
		tweet.text = html.unescape(tweet.text)
		return tweet

	def punctuation(self,tweet):
		"""
		Remove punctuation in the text of the tweet.
		"""
		tweet.text = "".join([char if char not in string.punctuation else "" for char in tweet.text])
		return tweet

	def expansion(self,tweet):
		"""
		Replace repeated characters up to 3 by the character once (e.g. amaaaaaaaaaaazing -> amazing / amaaazing -> amaaazing)
		Tried: then use Norvig's speller. (e.g. amaaaaaaaaaaazing -> amazing -> amazing / amaaazing -> amaaazing -> amazing) but computationally too long at the time
		"""
		tweet.text = re.compile(r'(\D)\1{3,}', re.IGNORECASE).sub(r'\1', tweet.text)
		# Method correct too long -> ~20,000 times longer with TextBlob().correct()
		# tweet.text = TextBlob(tweet.text).correct()
		return tweet

	def create_substitution_dictionary(self,url="https://www.netlingo.com/acronyms.php"):
		"""
		Create the dictionary for the substitutions checking using the netlingo acronyms list.
		"""
		html_content = requests.get(url).text

		soup = BeautifulSoup(html_content, 'lxml')
		d_substitution = {}
		for i in soup.div.contents[7].ul:
			d_substitution[i.a.get_text()] = i.div.get_text()

		return d_substitution
	
	def save_dictionary(self,dictionary,path_to_save="./"):
		"""
		Save the given dictionary to the given path in the pickel format.
		"""
		with open(path_to_save+'d_substitution.pkl', 'wb') as handle:
			pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return 0

	def load_dictionary(self,path_to_load="d_substitution.pkl"):
		"""
		Load a dictionary in pickle format using the given path.
		"""
		with open(path_to_load, 'rb') as handle:
			dictionary = pickle.load(handle)
		return dictionary


	def substitution(self,tweet,d_substitution):
		"""
		Replace substitutions into plain text.
		"""
		tweet.text = " ".join([d_substitution[word] if word in d_substitution else word for word in tweet.text.split()])
		return tweet

	def isHashtag(self,token):
		"""
		Return if the token is a hashtag or not.
		"""
		return token[0]=="#"

	def isMention(self,token):
		"""
		Return if the token is a mention or not.
		"""
		return token[0]=="@"

	def isRetweet(self,token):
		"""
		Return if the token is a mention or not.
		"""
		return token.lower()=="rt"


	def isUrl(self,token):
		"""
		Return if the token is an Url or not.
		"""
		if token.startswith("http:") or token.startswith("https:") or token.startswith("www"):
			return True
		return False
	
	
	def tweets_tokenize(self,tweet,replace=False):
		"""
		Replace tweets tokens by empty character if False, or by character special token if True (e.g. #Disaster -> '' if False / #Disaster -> HTG if True).
		"""
		if replace:
			tweet.text = " ".join([word if not self.isHashtag(word) else "htg" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isMention(word) else "mtn" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isUrl(word) else "url" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isRetweet(word) else "rtw" for word in tweet.text.split()])
		else:
			tweet.text = " ".join([word if not self.isHashtag(word) else "" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isMention(word) else "" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isUrl(word) else "" for word in tweet.text.split()])
			tweet.text = " ".join([word if not self.isRetweet(word) else "" for word in tweet.text.split()])
		return tweet

	def tweet_preprocess(self,tweet,d_substitution):
		"""
		Function for all tweet preprocessing execution.
		"""
		if self.normalizes:
			tweet = self.normalize(tweet)
		if self.tweets_tokens:
			tweet = self.tweets_tokenize(tweet,replace=True)
		if self.expansion:
			tweet = self.expansion(tweet)
		if self.substitutions:
			tweet = self.substitution(tweet,d_substitution)
		return tweet

def tweet_id(tweet):
	"""
	Retrieve the id of the tweet from the tweet json file.
	"""
	return tweet["id_str"]

def tweet_text(tweet):
	"""
	Retrieve the text of the tweet from the tweet json file.
	"""
	return tweet["full_text"]

def tweet_user_id(tweet):
	"""
	Retrieve the user id of the tweet from the tweet json file.
	"""
	if "id_str" in tweet["user"]:
		return tweet["user"]["id_str"]
	else:
		return str(tweet["user"]["id"])

def tweet_date(tweet):
	"""
	Retrieve the UTC date of the tweet from the tweet json file.
	"""
	if "created_at" in tweet:
		if "+0000" in tweet["created_at"]:
			return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["created_at"],'%a %b %d %H:%M:%S +0000 %Y'))
		else:
			return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["created_at"],'%b %d, %Y %H:%M:%S %p'))
	else:
		if "+0000" in tweet["createdAt"]:
			return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["createdAt"],'%a %b %d %H:%M:%S +0000 %Y'))
		else:
			return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["createdAt"],'%b %d, %Y %H:%M:%S %p'))

def tweet_date_local(tweet,zone="Etc/UTC"):
	"""
	Retrieve the local date of the tweet from the tweet json file.
	"""
	from_zone = tz.tzutc()
	to_zone = tz.gettz(zone)
	if "created_at" in tweet:
		if "+0000" in tweet["created_at"]:
			return datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["created_at"],'%a %b %d %H:%M:%S +0000 %Y')),'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(to_zone)
		else:
			return datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["created_at"],'%b %d, %Y %H:%M:%S %p')),'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(to_zone)
	else:
		if "+0000" in tweet["createdAt"]:
			return datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["createdAt"],'%a %b %d %H:%M:%S +0000 %Y')),'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(to_zone)
		else:
			return datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet["createdAt"],'%b %d, %Y %H:%M:%S %p')),'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(to_zone)

class Tweet(object):
	"""
	Each Tweet object represents a tweet.
	"""
	__slots__ = ("id","text","user_id","date","date_local")

	def __init__(self,tweet,zone="Etc/UTC"):
		"""
		The tweet object contains the tweet id, the text of the tweet, the id of the user who posted the tweet,\
		 and the dates (UTC and Local Time) the tweet was posted.
		List of the zones here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
		The tweet object also contains a field for the text preprocessed for extractive Oracle generation (stemmed, stopwords removed).
		"""
		self.id = tweet_id(tweet)
		self.text = tweet_text(tweet)
		self.user_id = tweet_user_id(tweet)
		self.date = tweet_date(tweet)
		self.date_local = tweet_date_local(tweet,zone)
	
TIME_ZONE = "Etc/UTC" # Choose the time zone of you event, list of time zones here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

preprocess = Preprocessing()
dictionary = preprocess.load_dictionary(path_to_load="d_substitution.pkl") # change the path of the "d_substitution.pkl" file released on our Github if needed

# Load tweets from json file:

PATH_TWEETS = "" # PATH of your json file

tweets = [] # contains all the tweets from the start of the stream to the considered time window

with open(PATH_TWEETS) as f:
		for line in tqdm(f, desc="Json to Tweet Object"):
			tweets.append(Tweet(json.loads(line),zone=TIME_ZONE))

tweets.sort(key = lambda t: t.date_local)

input_ids = []
attention_masks = []

for tweet in tweets:
	dbert_inps=dbert_tokenizer.encode_plus(preprocess.tweet_preprocess(tweet,dictionary).text,
											add_special_tokens = True,max_length=max_len,
											pad_to_max_length = True, return_attention_mask = True,truncation=True)
	input_ids.append(dbert_inps['input_ids'])
	attention_masks.append(dbert_inps['attention_mask'])

# Construct the vocabulary frequency for the considered time window
# Example of code to create it

time_window_frequencies_vector = [0]*(len(dbert_tokenizer.vocab)-1)

for tweet in input_ids:
	for token in tweet:
		if token!=0: # PAD token not used
			time_window_frequencies_vector[token-1]+=1


# Predict tweet salience score
# Example of code to compute it

model.compile(
  optimizer=tf.keras.optimizers.Adam(
	beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
	name='Adam'),
  loss=keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.MeanSquaredError(name="mean_squared_error")]
)

BATCH_SIZE = 128

# In this example, we consider the first increment, thus all the tweets require a salience score to be predict
salience_scores = model.predict([np.array(input_ids),np.array(attention_masks),np.array(time_window_frequencies_vector)],batch_size=BATCH_SIZE)


# Tweet selection
# Example of code to compute it


def compute_cos_score(gs,tweet):
	"""
	Compute the cosinus score between the gold standard text and the tweet text.
	"""
	X_list = word_tokenize(gs)
	Y_list = word_tokenize(tweet)

	l1 =[]
	l2 =[]

	X_set = set(X_list)
	Y_set = set(Y_list)

	rvector = X_set.union(Y_set)
	for w in rvector:
		if w in X_set: l1.append(1)
		else: l1.append(0)
		if w in Y_set: l2.append(1)
		else: l2.append(0)
	
	c = 0
	for i in range(len(rvector)):
		c+= l1[i]*l2[i]
	if float((sum(l1)*sum(l2))**0.5) != 0:
		cosine = c / float((sum(l1)*sum(l2))**0.5)
	else:
		cosine = 0
	return cosine

stemmer = nltk.stem.porter.PorterStemmer()
stop_words = set(stopwords.words('english')) 

def cleanPreProcessed(x):
	text = ''
	x=' '.join(x.split())
	x=re.sub(r'&amp;','&',x,flags=re.MULTILINE)
	x=html.unescape(x)
	punctuation = string.punctuation
	for y in x.split():
		if y.lower()!='rt' and y.lower()!='mt' and y.startswith('#')==False and y.startswith('@')==False and y.lower().startswith('http:')==False and y.lower().startswith('https:')==False and y not in punctuation and y.startswith('!')==False and y.startswith('.')==False and y.startswith('?')==False and y.lower() not in stop_words:
			text+=' '+ stemmer.stem(y)
	return text

def _surrogatepair(match):
	char = match.group()
	assert ord(char) > 0xffff
	encoded = char.encode('UTF-8')
	return (
		chr(int.from_bytes(encoded[:2], 'little')) + 
		chr(int.from_bytes(encoded[2:], 'little')))

def with_surrogates(text):
	return _nonbmp.sub(_surrogatepair, text)

_nonbmp = re.compile(r'[\U00010000-\U0010FFFF]')

threshold_salience = 0.2
set_threshold_similarity = 0.3
nb_max_tweets_per_window = 20

tweets_ordered_descending_salience_score = [x for y, x in sorted(zip(salience_scores, tweets),reverse=True)]
salience_scores_ordered_descending = [y for y, x in sorted(zip(salience_scores, tweets),reverse=True)]

increment_summary = []
len_summary = 0
threshold_similarity = set_threshold_similarity

# In this example, we consider only one increment

for i in range(len(tweets_ordered_descending_salience_score)):
	if len(increment_summary) < nb_max_tweets_per_window:
		if salience_scores_ordered_descending[i] > threshold_salience:
			keep = True
			for tweet_summary in increment_summary:
				if keep:
					if compute_cos_score(with_surrogates(cleanPreProcessed(tweet_summary)),with_surrogates(cleanPreProcessed(tweets_ordered_descending_salience_score[i].text)))>threshold_similarity:
						keep=False
		if keep:
			increment_summary.append(tweets_ordered_descending_salience_score[i].text)
			len_summary += len(word_tokenize(with_surrogates(cleanPreProcessed(tweets_ordered_descending_salience_score[i].text))))
			if len_summary > 50:
				threshold_similarity = set_threshold_similarity*(np.log(50)/np.log(len_summary))