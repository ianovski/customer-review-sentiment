import gensim
import logging
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from DataManipulation import DataManipulation

# The methods feature_vec_method(), get_avg_feature_vecs(), get_sentence_vectors() are written by:
# 	Title: Sentiment analysis using word2vec
# 	Author: Varun, D
# 	Date: 2018
# 	Code version: 1.0
# 	Availability: https://www.kaggle.com/varun08/sentiment-analysis-using-word2vec/

class RunWord2Vec:

	def run_word2vec(self,sentences,num_workers,dimension_size, min_count_size,window_size,sampling,file_name):
		"""Return word2vec model"""	
		model = gensim.models.Word2Vec(sentences,workers=num_workers,size=dimension_size, window=window_size, min_count=min_count_size, sample=sampling)
		# To make the model memory efficient
		model.init_sims(replace=True)
		model.save(file_name)
		return model

	def load_word2vec(self,path):
		"""Load saved word2vec model"""
		model = gensim.models.Word2Vec.load(path)
		return model

	def feature_vec_method(self, words, model, num_features):
		"""Average all word vectors in a paragraph"""
		# Pre-initialising empty numpy array for speed
		featureVec = np.zeros(num_features,dtype="float32")
		nwords = 0
		#Converting Index2Word which is a list to a set for better speed in the execution.
		index2word_set = set(model.wv.index2word)
		for word in  words:
			if word in index2word_set:
				nwords = nwords + 1
				featureVec = np.add(featureVec,model[word])

		# Dividing the result by number of words to get average
		featureVec = np.divide(featureVec, nwords)
		return featureVec

	def get_avg_feature_vecs(self, reviews, model, num_features):
		"""Calculating the average feature vector"""
		counter = 0
		reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
		for review in reviews:
		# Printing a status message every 1000th review
			if counter%1000 == 0:
				print("Review %d of %d"%(counter,len(reviews)))
            
			reviewFeatureVecs[counter] = self.feature_vec_method(review, model, num_features)
			counter = counter+1
        
		return reviewFeatureVecs

	def get_sentence_vectors(self,corpora,col_name,model,num_features):
		"""Return average feature vectors of a corpus"""
		data_manipulation = DataManipulation()
		clean_corpora = []
		for sentence in corpora[col_name]:
			clean_corpora.append(data_manipulation.review_wordlist(sentence, remove_stopwords=True))
		data_vecs = self.get_avg_feature_vecs(clean_corpora, model, num_features)
		return data_vecs
	
	# Title: Visualizing Word Vectors with t-SNE
	# Author: Delaney, J
	# Date: 2017
	# Code version: 3.0
	# Availability: https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
	def tsne_plot(self, model,plot_name): 
		"""2D t-SNE plot"""
		print("\nPlotting t-SNE...")
		labels = []
		tokens = []

		for word in model.wv.vocab:
			tokens.append(model[word])
			labels.append(word)
    
		tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
		new_values = tsne_model.fit_transform(tokens)

		x = []
		y = []
		for value in new_values:
			x.append(value[0])
			y.append(value[1])
        
		plt.figure(figsize=(25, 25)) 
		for i in range(len(x)):
			plt.scatter(x[i],y[i])
			plt.annotate(labels[i],
					xy=(x[i], y[i]),
					xytext=(5, 2),
					textcoords='offset points',
					ha='right',
					va='bottom')
		filename = 'figures/'+plot_name+'.png'
		plt.savefig(filename)
		plt.show()
