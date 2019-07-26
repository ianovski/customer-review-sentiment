from DataManipulation import DataManipulation
from RunWord2Vec import RunWord2Vec
from RunRandomForest import RunRandomForest
from ModelValidation import ModelValidation
from RunSVM import RunSVM

from gensim.models import word2vec
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Initialize classes
data_manip = DataManipulation()
run_word_2_vec = RunWord2Vec() # Clean up comments on bottom of page
run_random_forest = RunRandomForest() # Clean up comment at top of page
model_validation = ModelValidation()
run_svm = RunSVM()

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
data_manip.plot_word_frequency(sorted_wordcount)

# Importing saved training and testing files into memory
print("Importing split data....")
train = pd.read_csv('tweet_train_features2.csv')
train_labels = pd.read_csv('tweet_train_labels2.csv')
test = pd.read_csv('tweet_test_features2.csv')
test_labels = pd.read_csv('tweet_test_labels2.csv')

# Training word2vec model and visualizing with t-SNE
print("Training Word2Vec model and plotting t-SNE representation....")
model = run_word_2_vec.run_word2vec(sentences,num_workers,num_features,min_word_count,context,downsampling,"airlines_word2vec_model")
run_word_2_vec.tsne_plot(model, 'tsne_plot')

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
                      title='Random Forest Confusion Matrix, w/o Normalization')

# Plotting normalized confusion matrix of random forest model
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], forest_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='Random Forest Normalized Confusion Matrix')

# Printing classification report and accuracy score to console
print("Random Forest: ",classification_report(test_labels["airline_sentiment"],forest_output["airline_sentiment"]))  
print("Random Forest: ",accuracy_score(test_labels["airline_sentiment"],forest_output["airline_sentiment"]))  

# Training SVM Model
print("Training SVM Model....")
svm = run_svm.train_svm(train_data_vecs, train_labels,"airline_sentiment","rbf")

# Predicting test data with SVM model
print("Predicting SVM....")
svm_output = run_svm.predict_svm(svm,test_data_vecs,test,"id","airline_sentiment","svm_ouput.csv")

# Ploting non-normalized confusion matrix of SVM
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"],svm_output["airline_sentiment"], classes=["negative","neutral","positive"],
                      title='SVM Confusion Matrix, w/o Normalization')

# Plot normalized confusion matrix of SVM
model_validation.plot_confusion_matrix(test_labels["airline_sentiment"], svm_output["airline_sentiment"], classes=["negative","neutral","positive"], normalize=True,
                      title='SVM Normalized Confusion Matrix')

# Printing classification report and accuracy score of SVM model to console
print("SVM: ",classification_report(test_labels["airline_sentiment"],svm_output["airline_sentiment"]))  
print("SVM: ",accuracy_score(test_labels["airline_sentiment"],svm_output["airline_sentiment"]))
