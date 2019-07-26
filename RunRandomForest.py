import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RunRandomForest:


  def train_random_forest(self,training_data_vecs, train_labels,col_name,num_trees):
    """Return random forest model"""
    forest = RandomForestClassifier(n_estimators = num_trees)
    print("Fitting random forest to training data....")    
    forest = forest.fit(training_data_vecs, train_labels[col_name])
    return forest

  def predict_random_forest(self,model,test_vector,test_data,test_id_col,test_col_name,output_name):
    """Return output of random forest prediction and save to csv"""
    result = model.predict(test_vector)
    output = pd.DataFrame(data={"id":test_data["id"], test_col_name:result})
    output.to_csv(output_name, index=False, quoting=3 )
    return output