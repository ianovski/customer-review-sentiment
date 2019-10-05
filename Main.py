from DataManipulation import DataManipulation
from RunWord2Vec import RunWord2Vec
from RunRandomForest import RunRandomForest
from RunXGBoost import RunXGBoost
from RunANN import RunANN
from ModelValidation import ModelValidation
from RunSVM import RunSVM
from GoogleSentiment import GoogleSentiment
from IBMSentiment import IBMSentiment
from TextBlobSentiment import TextBlobSentiment
from MicrosoftSentiment import MicrosoftSentiment
from VaderSentiment import VaderSentiment
from AWSSentiment import AWSSentiment

from gensim.models import word2vec
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Initialize classes
data_manip = DataManipulation()
run_word_2_vec = RunWord2Vec() # Clean up comments on bottom of page
run_random_forest = RunRandomForest() # Clean up comment at top of page
run_xgb = RunXGBoost()
run_ann = RunANN()
model_validation = ModelValidation()
run_svm = RunSVM()
google_sentiment = GoogleSentiment()
ibm_sentiment = IBMSentiment()
tb_sentiment = TextBlobSentiment()
ms_sentiment = MicrosoftSentiment()
v_sentiment = VaderSentiment()
aws_sentiment = AWSSentiment()

# Initial parameters for word2vec
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Splitting dataset and saving files for training and testing
print("Splitting dataset....")
data_manip.split_dataset('Tweets.csv','airline_sentiment')
corpora = data_manip.get_csv_column('tweet_train_features2.csv',['text'])
sentences = data_manip.tokenize_sentences(corpora,'csv')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Removing stop words and visualizing word frequency of most used words
print("Visualizing words in corpora....")
full_corpora = data_manip.get_csv_column("Tweets.csv",["text"])
stop_words = ['get','co','http','2']
sorted_wordcount=data_manip.word_frequency(full_corpora,stop_words)
# data_manip.plot_word_frequency(sorted_wordcount)

# Importing saved training and testing files into memory
print("Importing split data....")
train = pd.read_csv('tweet_train_features2.csv')
train_labels = pd.read_csv('tweet_train_labels2.csv')
test = pd.read_csv('tweet_test_features2.csv')
test_labels = pd.read_csv('tweet_test_labels2.csv')

# Training word2vec model and visualizing with t-SNE
print("Training Word2Vec model and plotting t-SNE representation....")
model = run_word_2_vec.run_word2vec(sentences,num_workers,num_features,min_word_count,context,downsampling,"airlines_training_set_random_forest2")
# run_word_2_vec.tsne_plot(model, 'figures/word2vec_airline_training_set.png')

# Find average of word-vectors for each review
print("Finding average vector for each review....")
train_data_vecs = run_word_2_vec.get_sentence_vectors(train,'text',model,num_features)
test_data_vecs = run_word_2_vec.get_sentence_vectors(test,'text',model,num_features)

# Training Random forest model
print("Training Random Forest Model....")
forest = run_random_forest.train_random_forest(train_data_vecs, train_labels,"airline_sentiment",100)

# Predicting test data with Random Forest model
print("Predicting Random Forest....")
forest_output = run_random_forest.predict_random_forest(forest,test_data_vecs,test,"id","airline_sentiment","forest_ouput.csv")

# Printing results of confusion matrix to console
print("Plotting results....")
print(confusion_matrix(test_labels["airline_sentiment"],forest_output["airline_sentiment"],labels=["negative","neutral","positive"]))

# Plotting non-normalized confusion matrix of random forest model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],forest_output["airline_sentiment"], classes=["negative","neutral","positive"],
                      title='Random Forest Confusion Matrix')

# Plotting normalized confusion matrix of random forest model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], forest_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='Random Forest Normalized Confusion Matrix')

# Printing classification report and accuracy score to console
print("Random Forest: ",classification_report(test_labels["airline_sentiment"],forest_output["airline_sentiment"]))  
print("Random Forest: ",accuracy_score(test_labels["airline_sentiment"],forest_output["airline_sentiment"]))  

# Training XGBoost model
print("Training XGBoost Model....")
xgb = run_xgb.train_xgb(train_data_vecs, train_labels,"airline_sentiment",100)

# Predicting test data with Random Forest model
print("Predicting XGBoost....")
xgb_output = run_xgb.predict_xgb(xgb,test_data_vecs,test,"id","airline_sentiment","xgb_ouput.csv")

# Printing results of confusion matrix to console
print("Plotting results....")
print(confusion_matrix(test_labels["airline_sentiment"],xgb_output["airline_sentiment"],labels=["negative","neutral","positive"]))

# Plotting non-normalized confusion matrix of xgb model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],forest_output["airline_sentiment"], classes=["negative","neutral","positive"],
                      title='Random Forest Confusion Matrix')

# Plotting normalized confusion matrix of xgb model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], xgb_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='XGboost Normalized Confusion Matrix')

# Printing classification report and accuracy score to console
print("XGBoost: ",classification_report(test_labels["airline_sentiment"],xgb_output["airline_sentiment"]))  
print("XGboost: ",accuracy_score(test_labels["airline_sentiment"],xgb_output["airline_sentiment"]))  

# Training ANN
print("Training Artificial Neutral Network....")
ann = run_ann.train_ann(train_data_vecs, train_labels,"airline_sentiment")

# Predicting test data with ANN Model
print("Predicting ANN....")
ann_output = run_ann.predict_ann(ann,test_data_vecs,test,"id","airline_sentiment","ann_ouput.csv")

# Printing results of confusion matrix to console
print("Plotting results....")
print(confusion_matrix(test_labels["airline_sentiment"],ann_output["airline_sentiment"],labels=["negative","neutral","positive"]))

# Plotting non-normalized confusion matrix of ANN model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],ann_output["airline_sentiment"], classes=["negative","neutral","positive"],
                      title='ANN Confusion Matrix')

# Plotting normalized confusion matrix of ANN model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], ann_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='ANN Normalized Confusion Matrix')

# Printing classification report and accuracy score to console
print("ANN: ",classification_report(test_labels["airline_sentiment"],ann_output["airline_sentiment"]))  
print("ANN: ",accuracy_score(test_labels["airline_sentiment"],ann_output["airline_sentiment"]))  


# Training SVM Model
print("Training SVM Model....")
svm = run_svm.train_svm(train_data_vecs, train_labels,"airline_sentiment","rbf")

# Predicting test data with SVM model
print("Predicting SVM....")
svm_output = run_svm.predict_svm(svm,test_data_vecs,test,"id","airline_sentiment","svm_ouput.csv")

# Ploting non-normalized confusion matrix of SVM
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],svm_output["airline_sentiment"], classes=["negative","neutral","positive"],
                      title='SVM Confusion Matrix')

# Plot normalized confusion matrix of SVM
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], svm_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='SVM Normalized Confusion Matrix')

# Printing classification report and accuracy score of SVM model to console
print("SVM: ",classification_report(test_labels["airline_sentiment"],svm_output["airline_sentiment"]))  
print("SVM: ",accuracy_score(test_labels["airline_sentiment"],svm_output["airline_sentiment"]))

# Predict sentiment using Google sentiment analysis toolbox
print("Predicting Google sentiment....")
google_output = google_sentiment.run_sentiment(test)

model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], google_output["sentiment"], classes=["negative","neutral","positive"],
                      title='Google Sentiment Confusion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], google_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='Google Sentiment Normalized Confusion Matrix')
# # Printing classification report and accuracy score of google model
print("Google: ",classification_report(test_labels["airline_sentiment"],google_output["sentiment"]))  
print("Google: ",accuracy_score(test_labels["airline_sentiment"],google_output["sentiment"]))

# Predict sentiment using IBM sentiment analysis toolbox
print("Predicting IBM Watson sentiment....")
ibm_output = ibm_sentiment.run_sentiment(test)
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], ibm_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='IBM Watson Sentiment Normalized Confusion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], ibm_output["sentiment"], classes=["negative","neutral","positive"],
                      title='IBM Watson Sentiment Confusion Matrix')
# # Printing classification report and accuracy score of IBM model
print("IBM Watson: ",classification_report(test_labels["airline_sentiment"],ibm_output["sentiment"]))  
print("IBM Watson:",accuracy_score(test_labels["airline_sentiment"],ibm_output["sentiment"]))

print ("Predicting TextBlob sentiment....")
tb_output = tb_sentiment.run_sentiment(test)
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], tb_output["sentiment"], classes=["negative","neutral","positive"],
                      title='TextBlob Sentiment Confusion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], tb_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='TextBlob Sentiment Normalized Confusion Matrix')
# Printing classification report and accuracy score of TextBlob model
print("TextBlob: ",classification_report(test_labels["airline_sentiment"],tb_output["sentiment"]))  
print("TextBlob:",accuracy_score(test_labels["airline_sentiment"],tb_output["sentiment"]))


# print ("Predicting Microsoft Azure sentiment....")
ms_output = ms_sentiment.run_sentiment(test)
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], ms_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='Microsoft Sentiment Normalized Confusion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],ms_output["sentiment"], classes=["negative","neutral","positive"],
                      title='Microsoft Sentiment Confusion Matrix')
# Printing classification report and accuracy score of Microsoft model
print("Microsoft Azure: ",classification_report(test_labels["airline_sentiment"],ms_output["sentiment"]))  
print("Microsoft Azure:",accuracy_score(test_labels["airline_sentiment"],ms_output["sentiment"]))

print ("Predicting AWS sentiment....")
aws_output = aws_sentiment.run_sentiment(test)
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], aws_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='AWS Sentiment Normalized Confusion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],aws_output["sentiment"], classes=["negative","neutral","positive"],
                      title='AWS Sentiment Confusion Matrix')
# # Printing classification report and accuracy score of AWS model
print("AWS: ",classification_report(test_labels["airline_sentiment"],aws_output["sentiment"]))  
print("AWS:",accuracy_score(test_labels["airline_sentiment"],aws_output["sentiment"]))

print ("Predicting Vader sentiment....")
vs_output = v_sentiment.run_sentiment(test)
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], vs_output["sentiment"], classes=["negative","neutral","positive"],
                      title='Vader Sentiment Confustion Matrix')
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], vs_output["sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='Vader Sentiment Normalized Confusion Matrix')
# Printing classification report and accuracy score of Vader model
print("Vader: ",classification_report(test_labels["airline_sentiment"],vs_output["sentiment"]))  
print("Vader: ",accuracy_score(test_labels["airline_sentiment"],vs_output["sentiment"]))