"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement the perceptron and the training of the perceptron.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the perceptron model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the perceptron but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (perceptron) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produce as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Perceptron()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the perceptron. It is
up to your discretion on if you want to use them or add your own methods.
"""
import pandas as pd
import numpy as np
from nltk import stem, tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize
import math
import copy
from statistics import mean
import json
lemmatizer = WordNetLemmatizer()
def remove_stop(y, stopwords):
    new_list = []
    #print(y)
    for item in y:
        if item not in stopwords and not item[0].isdigit():
            new_list.append(item)
    return new_list

def cross_valid_train(headers, train_X, train_Y, lr, max_iter):
    weights = np.zeros((len(headers),))

    bias = 0
    for i in range(max_iter):
        count = 0
        #print("Epoch ", i)
        batch_size = 1
        for j in range(0, len(train_X), batch_size):
            rows = train_X[j]
            label = np.matmul(rows, weights) + bias
            if(label<0):
                label = -1
            else:
                label = 1
            model_result = np.squeeze(label)
            #print(rows)
            expected_result = train_Y[j]
            loss = model_result - expected_result
            #print("HERE")
            weights = weights - loss * lr * rows
            bias = bias - loss*lr
    return weights


class Perceptron():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the perceptron as instance attributes.
        """
        self.weights = None
        self.bias = None
        self.max_iter = None
        self.learning_rate = None
        self.vocabulary = None
        self.doc_count = None
        
    def feature_extraction(self, raw_data, use_language_vocab=False):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in perceptron training.
        """
        text = raw_data['Text']
        #new_text = text.apply()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r"\w+")
        """
        special_chars = ['\(', '\)', '.', ',']
        
        text = text.replace("=+", '', regex=True)
        text = text.replace("\`+", '', regex=True)
        text = text.replace("\~+", '', regex=True)
        text= text.replace("\^+", '', regex=True)
        text = text.replace("\\", '')
        text = text.replace("\[+", '', regex=True)
        text = text.replace("\]+", '', regex=True)
        text = text.replace("\>+", ' ', regex=True)
        text = text.replace('!+', '', regex=True)
        text = text.replace('\(', '', regex=True)
        text = text.replace('\)', '', regex=True)
        text = text.replace("'", '', regex=True)
        text = text.replace('"', '', regex=True)
        text = text.replace(",", '', regex=True)
        text = text.replace("-", ' ', regex=True)
        text = text.replace("_", ' ', regex=True)
        text = text.replace("/", ' ', regex=True)
        text = text.replace("\|+", ' ', regex=True)
        #text = text.replace("||", ' ', regex=True)
        text = text.replace("{", '', regex=True)
        text = text.replace("}", '', regex=True)
        text = text.replace("+", '', regex=False)
        text = text.replace("@", '', regex=True)
        text = text.replace("#", '', regex=True)
        text=  text.replace("&", '', regex=True)
        text = text.replace("%", '', regex=True)
        text = text.replace("\*", '', regex=True)
        text = text.replace("$", '')
        text =text.replace("\.+", '', regex=True)
        text = text.replace(":+", ' ', regex=True)
        text = text.replace(";+", ' ', regex=True)
        text = text.replace("\?+", ' ', regex=True)
        """
        #text = text.replace("^[a-zA-Z0-9 ]+", '', regex=True)
        #print(text[0])
        #exit()
        tokenizer = RegexpTokenizer(r'\w+')
        stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own',
                     'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
                     'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
                     'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
                     'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                     'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                     'which','those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
                     'here', 'than']
        
        new_text = text.apply(word_tokenize)
        new_text = new_text.apply(lambda x: remove_stop(x, stopwords))

        
        #new_text = new_text.apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
        new_text = new_text.apply(lambda x: [stemmer.stem(y) for y in x])
        #print(new_text[0])
        #exit()
        language_vocab = {}
        
        terms_per_doc = {}
        count = 0

        for document in new_text:
            terms_per_doc[count] = {}
            for word in document:
                if word not in terms_per_doc[count].keys():
                    terms_per_doc[count][word] = 1
                else:
                    terms_per_doc[count][word]+=1
            count+=1
            for word in terms_per_doc[count-1].keys():
                if word not in language_vocab:
                    language_vocab[word] = 1
                else:
                    language_vocab[word] +=1
        if(use_language_vocab == True):
            language_vocab = self.vocabulary
        else:
            self.vocabulary = language_vocab
        headers = sorted(list(language_vocab.keys()))
        
        #print(headers)
        #print(len(headers))
        #exit()
        word_loc_in_headers = {}
        for i in range(len(headers)):
            word_loc_in_headers[headers[i]] = i

        TF_IDF_DF = np.asarray([headers])
        count = 0

        attrib_count = len(headers)
        num_elements = len(new_text)
        vector = np.zeros((num_elements, attrib_count))
        num_docs = max(terms_per_doc.keys())
        self.doc_count = num_docs
        for doc in new_text:
            
            
            word_ind = 0
            for word in doc:
                try:
                    tf = terms_per_doc[count][word]
                    tf = math.log(1 + tf)
                    doc_occ = language_vocab[word]
                
                    idf = math.log(num_docs/(1+doc_occ))+1
                    tf_idf = tf * idf
                
                    vector_index = word_loc_in_headers[word]
                    vector[count][vector_index] = tf_idf
                except:
                    continue

            count +=1


        labels =  copy.copy(np.squeeze(np.asarray(raw_data['Label'])))

        return headers, vector, labels

    def sgn_function(self, perceptron_input):
        """
        Optional helper method to code the sign function for the perceptron.
        """
        #perceptron_input = 1/(1 + np.exp(-1 * perceptron_input))

        if(perceptron_input<0):
                return -1
        return 1

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights of the perceptron.
        """
        self.weights = new_weights
        return

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias of the perceptron.
        """
        self.bias = new_bias
        return

    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point.
        """
        
        return np.matmul(data_point, self.weights) + self.bias

    def train(self, labeled_data, learning_rate=None, max_iter=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and max number of iterations. Once you find the
        optimal values of the hyperparameters, update the default values for each keyword argument to reflect those
        values.

        The goal of this function is to train the perceptron on the labeled data. Feel free to code this however you
        want.
        """
        if(learning_rate !=None and max_iter !=None):
            self.max_iter = max_iter
            self.learning_rate = learning_rate
            headers, train_X, train_Y = self.feature_extraction(labeled_data)
            self.weights = np.zeros((len(headers),))
            for i in range(len(train_Y)):
                if(train_Y[i] == 0):
                    train_Y[i] = -1
            self.bias = 0
            for i in range(5):
                count = 0
                #print("Epoch ", i)
                batch_size = 1
                for j in range(0, len(train_X), batch_size):
                    rows = train_X[j]
                    labels = self.predict_labels(rows)
                    model_result = np.squeeze(self.sgn_function(labels))
                    #print(rows)
                    expected_result = train_Y[j]
                    loss = model_result - expected_result
                    #print("HERE")
                    if(model_result != expected_result):
                        self.weights = self.weights - model_result * learning_rate * rows
                        self.bias = self.bias - loss *learning_rate
                    #print("Epoch ", i, " Batch ", j//20)
        else:
            #K - FOLD CROSS VALIDATION
            print("YOU ARE ENTERING K-FOLD VALIDATION BECAUSE YOU DIDN'T PROVIDE HYPERPARAMS")
            headers, train_X, train_Y = self.feature_extraction(labeled_data)
            for i in range(len(train_Y)):
                if(train_Y[i] == 0):
                    train_Y[i] = -1
            data_len = len(train_X)
            block_size = int(data_len/5)
            train_x_blocks = []
            train_y_blocks = []
            for i in range(0, data_len, block_size):
                train_x_blocks.append(train_X[i:i+block_size])
                train_y_blocks.append(train_Y[i:i+block_size])
            lr_values = [0.001, 0.01, 0.05, 0.1]
            max_iter_values = [1, 2, 3, 5, 8, 10]
            accuracy_grid = {}
            for i in lr_values:
                accuracy_grid[i] = {}
                for j in max_iter_values:
                    accuracy_grid[i][j] = []
                    print("MAX_ITER = ", j, " LR = ", i)
                    for k in range(5):
                        validation_x = train_x_blocks[k]
                        validation_y = train_y_blocks[k]
                        new_train = copy.copy(train_x_blocks)
                        new_train_label = copy.copy(train_y_blocks)
                        new_train.pop(k)
                        new_train_label.pop(k)
                        new_train = np.concatenate(new_train, axis=0)
                        new_train_label = np.concatenate(new_train_label, axis=0)
                        weights = cross_valid_train(headers, new_train, new_train_label, i, j)
                        predictions = np.squeeze(np.matmul(validation_x, weights))
                        #print(predictions)
                        count = 0
                        for l in range(len(predictions)):
                            if((predictions[l]>=0 and validation_y[l] == 1) or (predictions[l]<0 and validation_y[l] == -1)):
                                count+=1
                        accuracy_grid[i][j].append(count/len(predictions))
                    accuracy_grid[i][j] = mean(accuracy_grid[i][j])
                    print(weights)
            with open('perceptron.json', 'w') as json_file:
                json.dump(accuracy_grid, json_file)
                        
        #predictions = np.squeeze(np.matmul(train_X, self.weights))
        #count = 0
        """
        for i in range(len(predictions)):
            if((predictions[i]>0 and train_Y[i] == 1) or (predictions[i]<=0 and train_Y[i] == -1)):
                count +=1
        print("Training Accuracy: ", count/len(predictions))
        """
        return

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. The first input is the data in the form of a 
        pandas dataframe. 
        
        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers that are either 0 (negative) or 1
        (positive) for each data point.
        
        The rest of the implementation can be fully customized.
        """
        headers, vector, labels = self.feature_extraction(data, True)
        predictions = np.squeeze(np.matmul(vector, self.weights))
        
        predicted_labels = list(predictions)
        for i in range(len(predicted_labels)):
            if(predicted_labels[i]<0):
                predicted_labels[i] = 0
            else:
                predicted_labels[i] = 1
        return predicted_labels
