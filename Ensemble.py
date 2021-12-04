"""
You only need to implement bagging.
"""
from statistics import mean
import pandas as pd
from Logistic import Logistic
from scipy.stats import mode
import numpy as np
class Ensemble():
    def __init__(self):
        """
        You may initialize the parameters that you want and remove the 'return'
        """
        self.model_weights = []
        self.classifiers = []
        return
    
    def feature_extraction(self):
        """
        Use the same method as in Logistic.py
        """
        return
    
    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        return

    def train(self, labeled_data, num_clf=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.

        There is no limitation on how you implement the training process.
        """
        data_size = len(labeled_data)
        data_size = data_size//num_clf

        for i in range(num_clf):

            sample = labeled_data.sample(frac=1, replace=True)
            sample.reset_index(inplace=True, drop=True)
            classifier = Logistic()
            classifier.train(sample, learning_rate=0.001, max_epochs=10, feature_method='bigram', reg_method='L2', lam=0.001)
            self.model_weights.append(classifier.weights)
            self.classifiers.append(classifier)
        return

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The only input is the data in the 
        form of a pandas dataframe. 

        Finally, you must return the variable predicted_labels which should contain a list of all the predicted 
        labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 
        1 (positive) for each data point.

        The rest of the implementation can be fully customized.
        """
        for i in range(len(self.classifiers)):
            headers, vector, labels = self.classifiers[i].feature_extraction(data, use_language_vocab = True, method=self.classifiers[i].cur_method)
            print(vector.shape)
            print(np.asarray(self.classifiers[i].weights).shape)
            #print(self.classifiers[i].weights)
            predictions = np.squeeze(self.classifiers[i].predict_labels(0, np.matmul(vector, self.classifiers[i].weights), True))
            predicted_labels_i = list(predictions)
            print(np.asarray(predicted_labels_i))
            for j in range(len(predicted_labels)):
                if(predicted_labels_i[j]<0.5):
                    predicted_labels_i[j] = 0
                else:
                    predicted_labels_i[j] = 1
            count = 0
            for j in range(len(predicted_labels_i)):
                if(predicted_labels_i[j] == data['Label'][j]):
                    count+=1
            print("Classifier ", i, ' Accuracy = ', count/len(data))
            predicted_labels.append(predicted_labels_i)
        predicted_labels = mode(predicted_labels, axis=0)[0][0]
        count = 0
        for j in range(len(predicted_labels)):
            if(predicted_labels[j] == data['Label'][j]):
                count +=1
        print("Ensemble Learner Accuracy ", count/len(data))
        print(predicted_labels)
        exit()
        
        
data = pd.read_csv('data.csv')
logistic_reg = Ensemble()
logistic_reg.train(data, num_clf=10)


predictions = logistic_reg.predict(data)
#count = 0
#for i in range(len(predictions)):
#    if(predictions[i] == data['Label'][i]):
#        count+=1
#print("Training Accuracy :", count/len(predictions))
            
    

