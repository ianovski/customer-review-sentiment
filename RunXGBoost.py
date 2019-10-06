from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class RunXGBoost():
	def train_xgb(self,training_data_vecs,train_labels,col_name,num_trees):
		"""Return XGBoost model"""
		print("Fitting XGBoost to training data....")    
		xgb = XGBClassifier(n_estimators=num_trees)
		xgb.fit(training_data_vecs,train_labels[col_name])
		return xgb

	def predict_xgb(self,model,test_vector,test_data,test_id_col,test_col_name,output_name):
		"""Return output of XGBoost and save to csv"""
		result = model.predict(test_vector)
		output = pd.DataFrame(data={"id":test_data["id"], test_col_name:result})
		output.to_csv(output_name, index=False, quoting=3 )
		return output
