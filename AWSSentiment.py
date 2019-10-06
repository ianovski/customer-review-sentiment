import boto3
import pandas as pd
import time

class AWSSentiment(): 

	def __init__(self):
		self.client = boto3.client('comprehend',
		region_name="us-east-2",
		aws_access_key_id = "ENTER_ACCESS_ID",
		aws_secret_access_key = "ENTER_ACCESS_KEY")


	def get_sentiment(self,client,document):
		"""Run AWS API and get sentiment for given sentence"""
		response = client.detect_sentiment(
			Text=document,
			LanguageCode='en'
		)
		sentiment = response["Sentiment"]
		sentiment = sentiment.lower()
		if(sentiment=="mixed"):
			sentiment="neutral"
		return(sentiment)

	def save_results(self,test_data,sentiments):
		"""Save sentiments and IDS to CSV"""
		output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
		output.to_csv("aws_sentiment.csv", index=False, quoting=3 )
		return output

	def run_sentiment(self,test_data):
		""" Run AWS Sentiment analysis"""
		sentences = test_data['text']
		sentiments = []
		count = 0
		for sentence in sentences:
			sentiments.append(self.get_sentiment(self.client,sentence))
			count+=1
			if(count%20==0):
				print("AWS analyzing review #", count)
				time.sleep(1)
		return self.save_results(test_data,sentiments)


