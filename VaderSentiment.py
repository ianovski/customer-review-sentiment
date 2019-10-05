from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

class VaderSentiment():

	def save_results(self,test_data,sentiments):
		output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
		output.to_csv("vader_sentiment.csv", index=False, quoting=3 )
		return output
	
	def get_sentiment(self,sentence,analyzer):
		negative_threshold = -0.05
		positive_threshold = 0.05
		sentiment_score = analyzer.polarity_scores(sentence)
		sentiment_classification = "negative" if sentiment_score['compound']<=negative_threshold \
			else "neutral" if sentiment_score['compound']<positive_threshold \
			else "positive"
		return(sentiment_classification)

	def run_sentiment(self,test_data):
		analyzer = SentimentIntensityAnalyzer()
		sentences = test_data['text']
		sentiments = []
		count = 0
		for sentence in sentences:
			sentiments.append(self.get_sentiment(sentence,analyzer))
			count+=1
			if(count%100==0):
				print("Vader analyzing review #", count)	
		return self.save_results(test_data,sentiments)


# vader_sentiment = VaderSentiment()
# test_labels = pd.read_csv('tweet_test_features2_sample.csv')
# print(vader_sentiment.run_sentiment(test_labels))
