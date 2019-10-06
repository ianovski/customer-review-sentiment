from textblob import TextBlob
import pandas as pd

class TextBlobSentiment():

	def get_sentiment(self,text):
		negative_threshold = -0.33
		neutral_threshold = 0.33
		text_blob = TextBlob(text)
		sentiment = text_blob.sentiment.polarity
		sentiment_classification = "negative" if sentiment<=negative_threshold \
			else "neutral" if sentiment<=neutral_threshold \
			else "positive"
		return sentiment_classification

	# This should be generalized and in a different file
	'''Save results to csv and return Panda dataframe'''
	def save_results(self,test_data,sentiments):
		output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
		output.to_csv("tb_sentiment.csv", index=False, quoting=3 )
		return output

	def run_sentiment(self,dataset):
		sentences = dataset['text']
		sentiments = []
		for sentence in sentences:
			sentiments.append(self.get_sentiment(sentence))
		return self.save_results(dataset,sentiments)


# test_labels = pd.read_csv('tweet_test_features2_sample.csv')
# tb_sentiment = TextBlobSentiment()
# sentiments = tb_sentiment.run_sentiment(test_labels)
# print("FINAL:", sentiments)