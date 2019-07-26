import pandas as pd
from sklearn.svm import SVC

class RunSVM:
  def train_svm(self,training_data_vecs, train_labels,col_name,kernel_type):
    """Return SVM model"""
    svm = SVC(kernel=kernel_type,gamma='auto')
    print("Fitting SVM to training data....")    
    svm = svm.fit(training_data_vecs, train_labels[col_name])
    return svm

  def predict_svm(self,model,test_vector,test_data,test_id_col,test_col_name,output_name):
    """Return output of SVM prediction and save to csv"""
    result = model.predict(test_vector)
    output = pd.DataFrame(data={"id":test_data["id"], test_col_name:result})
    output.to_csv(output_name, index=False, quoting=3 )
    return output