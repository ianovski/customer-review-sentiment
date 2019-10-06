########### Python 3.2 #############
import http.client
import urllib.request, urllib.parse, urllib.error
import base64
import json
import pandas as pd
import time

class MicrosoftSentiment():
    def get_headers(self):
        """Compile header for REST API"""
        headers = {
            # Request headers
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': 'ENTER_SUBSCRIPTION_KEY',
        }
        return headers

    def get_params(self):
        """Compile params of REST API"""
        params = urllib.parse.urlencode({
            # Request parameters
            'showStats': '{boolean}',
        })
        return params

    def get_documents(self,sentence):
        """Convert sentence to document"""
        documents = {
            "documents": [
            {
                "language": "en",
                "id": "1",
                "text": sentence
            }
            ]
        }
        return documents

    def get_sentiment(self,documents,params,headers):
        """Post document to MSFT Rest API and get sentiment score. Convert score to sentiment category"""
        negative_threshold = 0.33
        neutral_threshold = 0.67
        json_data = json.dumps(documents)
        sentiment_classification = "neutral"
        try:
            conn = http.client.HTTPSConnection('eastus.api.cognitive.microsoft.com')
            conn.request("POST", "/text/analytics/v2.1/sentiment?%s" % params, json_data, headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            encoding = response.info().get_content_charset('utf8')  # JSON default
            json_result = json.loads(data.decode(encoding))
            sentiment_score = json_result['documents'][0]['score']
            sentiment_classification = "negative" if sentiment_score<=negative_threshold \
                else "neutral" if sentiment_score<=neutral_threshold \
                else "positive"
        except Exception as e:
            print("[Error]",)
        return(sentiment_classification)

    def save_results(self,test_data,sentiments):
        """Save sentiments and IDs to CSV"""
        output = pd.DataFrame(data={"id":test_data["id"], "sentiment":sentiments})
        output.to_csv("microsoft_sentiment.csv", index=False, quoting=3 )
        return output

    def run_sentiment(self,test_data):
        """Run MSFT Azure Sentiment analysis model"""
        sentences = test_data['text']
        sentiments = []
        count = 0
        for sentence in sentences:
            sentiments.append(self.get_sentiment(self.get_documents(sentence),self.get_params(),self.get_headers()))
            count+=1
            if(count%100==0):
                print("Microsoft analyzing review #", count)
                print("Pausing Microsoft API requests for 1 minute.....") # API requests limited to 600/minute
                time.sleep(65)
        return self.save_results(test_data,sentiments)
