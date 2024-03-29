# Autogenerated service credentials:
'''{
  "apikey": "nYfl1v4YN5_sAoT89gggtyCMBCL4-dRtsURk-WsCFkKh",
  "iam_apikey_description": "Auto-generated for key 2c9fb241-57f9-4bb2-9bbc-b3011a07ab0c",
  "iam_apikey_name": "Auto-generated service credentials",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/92b56f4044da278e919b81ec8ebcfc90::serviceid:ServiceId-c303e5ba-8b06-4ad0-ab96-629c252d4196",
  "url": "https://gateway.watsonplatform.net/natural-language-understanding/api"
}
'''
from __future__ import print_function
import requests
import pandas as pd
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

class IBMSentiment():
	def __init__(self):
		self.service = NaturalLanguageUnderstandingV1(
			version='2018-03-16',
			url='https://gateway.watsonplatform.net/natural-language-understanding/api',
			iam_apikey='ENTER_API_KEY')

	def get_sentiment(self, service,sentence):
		"""Run IBM API and get sentiment for given sentence"""
		try:
			sentiment=SentimentOptions()
			response = service.analyze(
			text=sentence,
			features=Features(sentiment=SentimentOptions())
			).get_result()
		except Exception as e:
			print(e.message)
			return "neutral" # If unable to categorize, set sentiment to "Neutral"
		return response['sentiment']['document']['label']

	def save_results(self,test_data,sentiments):
		"""Save sentiments and IDs to CSV"""
		output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
		output.to_csv("ibm_sentiment.csv", index=False, quoting=3 )
		return output

	def run_sentiment(self,test_data):
		""" Run IBM Sentiment analysis"""
		sentences = test_data['text']
		sentiments = []
		count = 0
		for sentence in sentences:
			sentiments.append(self.get_sentiment(self.service,sentence))
			count+=1
			if(count%100==0):
				print("IBM Watson analyzing review #", count)
		return self.save_results(test_data,sentiments)