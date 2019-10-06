# Setup guide for using Google libraries: 
# 	https://cloud.google.com/natural-language/docs/quickstart-client-libraries#client-libraries-install-python
# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ModelValidation import ModelValidation
model_validation = ModelValidation()

class GoogleSentiment:

	"""Instantiates a Google client. Replace with proper init statement"""
	def __init__(self):
		self.client = language.LanguageServiceClient()

	def parse_text(self,text):
		"""Convert sentence to document"""
		document = types.Document(
			content=text,
			type=enums.Document.Type.PLAIN_TEXT)
		return document

	def get_sentiment(self,client,document):
		"""Run Google API and get sentiment for given sentence"""
		negative_threshold = -0.33
		neutral_threshold = 0.33
		sentiment = client.analyze_sentiment(document=document).document_sentiment
		sentiment_classification = "negative" if sentiment.score<=negative_threshold \
			else "neutral" if sentiment.score<=neutral_threshold \
			else "positive"
		return sentiment_classification

	def get_data(self,client,sentences):
		"""Get sentiments of entire dataset"""
		sentiments = []
		counter = 0
		for sentence in sentences:
			sentiments.append(self.get_sentiment(client,self.parse_text(sentence)))
			counter+=1
			if(counter%100==0):
				print("Test set index = ", counter, "......")
			if(counter%598==0):
				print("Pausing Google API requests for 1 minute.....") # API requests limited to 600/minute
				time.sleep(65)
		return sentiments

	def save_results(self,test_data,sentiments):
		"""Save sentiments and IDs to CSV"""
		output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
		output.to_csv("google_sentiment.csv", index=False, quoting=3 )
		return output

	def run_sentiment(self,dataset):
		"""Run Google Sentiment analysis"""
		sentences = dataset['text']
		sentiments = self.get_data(self.client,sentences)
		return self.save_results(dataset,sentiments)
		
