import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=FutureWarning)
	import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from numpy import argmax
from numpy import array
from numpy import amax
from numpy import where

class RunANN():

	def encode_vars_int(self,train_labels):
		""" Encode string categories as integers """
		values = array(train_labels)
		label_encoder = LabelEncoder()
		int_encoded = label_encoder.fit_transform(values)
		return int_encoded
	
	def encode_vars_bin(self,int_encoded):
		""" Encode integers as binary categories """
		bin_encoded = np_utils.to_categorical(int_encoded)
		return bin_encoded

	def decode_vars_bin(self,encoded_labels):
		""" Decode binary categories into string categories """
		decoded_vars = []
		for encoded in encoded_labels:
			maxIndex = (where(encoded == amax(encoded)))[0]
			maxIndex = maxIndex[0]
			decoded_var = "negative" if maxIndex == 0 else "neutral" if maxIndex == 1 else "positive"
			decoded_vars.append(decoded_var)
		return decoded_vars


	def train_ann(self,training_data_vecs,train_labels,col_name):
		"""Train and return ANN model"""
		print("Fitting ANN to training data....")
		dummy_labels = self.encode_vars_bin(self.encode_vars_int(train_labels[col_name]))
		# Initilizing the ANN
		ann = Sequential()
		# Adding the input layer and the first hidden layer
		ann.add(Dense(activation="relu", input_dim=300, units=150, kernel_initializer="uniform")) # Double chekc output_dim and input_dim (don't hard-code values. calculate from number of dimensions in the model)
		# Adding the 2nd hidden layer
		ann.add(Dense(activation="relu", units=150, kernel_initializer="uniform")) # Double output_dim (don't hard-code values. calculate from number of dimensions in the model)
		# Adding the output layer
		ann.add(Dense(activation="sigmoid", units=3, kernel_initializer="uniform")) 
		# Compiling the ANN
		ann.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
		# encode class values as integers
		ann.fit(training_data_vecs,dummy_labels,batch_size=10,epochs=200)
		return ann

	def predict_ann(self,model,test_vector,test_data,test_id_col,test_col_name,output_name):
		"""Return output of ANN and save to csv"""
		result = model.predict(test_vector)
		result_decoded = self.decode_vars_bin(result)	
		output = pd.DataFrame(data={"id":test_data["id"], test_col_name:result_decoded})
		output.to_csv(output_name, index=False, quoting=3 )
		return output
