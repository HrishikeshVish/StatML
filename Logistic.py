"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the logistic model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the logistic function but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (logistic) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produces as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the logistic function. 
It is up to your discretion on if you want to use them or add your own methods.
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
import time
lemmatizer = WordNetLemmatizer()
def remove_stop(y, stopwords):
    new_list = []
    #print(y)
    for item in y:
        if item not in stopwords and not item[0].isdigit():
            new_list.append(item)
    return new_list

def regularizer_cv(method, lam, weights):
    if method == 'L1':
        sign_matrix = np.where(weights >=0, 1, 0)
        return lam * sign_matrix
    if method == 'L2':
        return lam * weights

def cross_valid_train(headers, train_X, train_Y, lr, max_iter, reg_method, lam):
    weights = np.zeros((len(headers),))

    bias = 0
    for i in range(max_iter):
        count = 0
        print("Epoch ", i)
        batch_size = 1
        for j in range(0, len(train_X), batch_size):
            rows = train_X[j]
            prod = np.matmul(rows, weights) + bias
            if(prod.any()<0):
                label = np.exp(prod)/(1 + np.exp(prod))
            else:
                label = 1/(1 + np.exp(-1*prod))
            model_result = np.squeeze(label)
            #print(rows)
            expected_result = train_Y[j]
            loss = model_result - expected_result
            #print("HERE")
            weights = weights - loss * lr * rows
            weights = weights - lr * (loss * rows + regularizer_cv(reg_method, lam, weights))
            bias = bias - loss*lr
                    
    return weights
def getVectorSpaceUnigram(new_text):
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
        headers = sorted(list(language_vocab.keys()))
        return headers, language_vocab, terms_per_doc
def getVectorSpaceBigram(new_text):
    language_vocab = {}
    bigrams_corpus = {}
    terms_per_doc = {}

    count = 0
    for document in new_text:
        
        for i in range(len(document)-1):
            word = document[i]
            next_word = document[i+1]
            if (word, next_word) not in bigrams_corpus.keys():
                bigrams_corpus[(word, next_word)] = 1
            else:
                bigrams_corpus[(word, next_word)] += 1
            if word not in terms_per_doc.keys():
                if(word not in language_vocab):
                    language_vocab[word] = 1
                else:
                    language_vocab[word] +=1
                terms_per_doc[word] = 1
            else:
                terms_per_doc[word]+=1
            if (i == len(document)-2):
                if next_word not in terms_per_doc.keys():
                    if(next_word not in language_vocab):
                        language_vocab[next_word] = 1
                    else:
                        language_vocab[next_word] +=1
                       
                    terms_per_doc[next_word] = 1
                else:
                    terms_per_doc[next_word] += 1
    headers = sorted(list(language_vocab.keys()))
    return headers, language_vocab, terms_per_doc, bigrams_corpus
        
    
def getVectorSpaceTrigram(new_text):
    language_vocab = {}
    trigrams_corpus = {}
    terms_per_doc = {}

    count = 0
    for document in new_text:
        
        for i in range(len(document)-2):
            word = document[i]
            next_word = document[i+1]
            later_word = document[i+2]
            if (word, next_word, later_word) not in trigrams_corpus.keys():
                trigrams_corpus[(word, next_word, later_word)] = 1
            else:
                trigrams_corpus[(word, next_word, later_word)] += 1
            if word not in terms_per_doc.keys():
                if(word not in language_vocab):
                    language_vocab[word] = 1
                else:
                    language_vocab[word] +=1
                terms_per_doc[word] = 1
            else:
                terms_per_doc[word]+=1
            if (i == len(document)-2):
                if next_word not in terms_per_doc.keys():
                    if(next_word not in language_vocab):
                        language_vocab[next_word] = 1
                    else:
                        language_vocab[next_word] +=1
                       
                    terms_per_doc[next_word] = 1
                else:
                    terms_per_doc[next_word] += 1
    headers = sorted(list(language_vocab.keys()))

    return headers, language_vocab, terms_per_doc, trigrams_corpus


class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the logistic function as instance
        attributes.
        """
        self.weights = None
        self.bias = None
        self.max_iter = 500
        self.learning_rate = 0.01
        self.vocabulary = None
        self.cur_method = None
        
    def feature_extraction(self, raw_data, use_language_vocab=False, method=None):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in training. You need to implement unigram, bigram and trigram.
        """
        text = raw_data['Text']
        #new_text = text.apply()
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r"\w+")
        stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own',
                     'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
                     'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
                     'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
                     'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                     'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                     'which','those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
                     'here', 'than']
        
        special_chars = ["=+", "\`+", "\~+", "\^+", "\[+", "\]+", "\>+", '!+', '\(', '\)', "'", '"', ",", "-", "_", "/", "\|+", "||", "{", "}", "@", "#", "&",
                         "%", "\*", "\.+", ":+", ";+", "\?+"]
        for char in special_chars:
            text = text.replace(char, '', regex=True)
        
        text = text.replace("\\", '')
        text = text.replace("+", '', regex=False)
        text = text.replace("$", '')
        
        new_text = text.apply(word_tokenize)
        new_text = new_text.apply(lambda x: remove_stop(x, stopwords))
        #new_text = new_text.apply(lambda x: [stemmer.stem(y) for y in x])
        
        if method == 'unigram':
            self.cur_method = 'unigram'
            headers, vector_space, terms_per_doc = getVectorSpaceUnigram(new_text)
            
            word_loc_in_headers = {}
            
            if(use_language_vocab == True):
                vector_space = self.vocabulary
            else:
                self.vocabulary = vector_space
            headers = sorted(list(vector_space.keys()))
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
                        doc_occ = vector_space[word]
                        idf = math.log(num_docs/(1+doc_occ)) +1
                        tf_idf = tf * idf
                        vector_index = word_loc_in_headers[word]
                        vector[count][vector_index] = tf_idf
                    except:
                        continue
                count +=1
            

        if method == 'bigram':
            self.cur_method='bigram'
            headers, vector_space,terms_per_doc, bigram_corpus = getVectorSpaceBigram(new_text)
            
            word_loc_in_headers = {}
            
            if(use_language_vocab == True):
                vector_space = self.vocabulary
            else:
                self.vocabulary = vector_space
            headers = sorted(list(vector_space.keys()))
            for i in range(len(headers)):
                word_loc_in_headers[headers[i]] = i
            TF_IDF_DF = np.asarray([headers])
            count = 0
            attrib_count = len(headers)
            num_elements = len(new_text)
            vector = np.zeros((num_elements, attrib_count))
            num_docs = num_elements
            self.doc_count = num_docs
            for doc in new_text:
                word_ind = 0
                for i in range(len(doc)):
                    try:
                        p_i = terms_per_doc[doc[i]]/attrib_count
                        laplace_factor = 1e-6
                        if(i == 0 and i<len(doc)-1):

                            p_i_next = (bigram_corpus[(doc[i], doc[i+1])]+0.00001)/((terms_per_doc[doc[i]]/attrib_count) + laplace_factor)

                            tf = p_i * p_i_next * (1/p_i)
                        elif(i == len(doc) - 1):

                            p_i_prev = (bigram_corpus[(doc[i-1], doc[i])]+0.00001)/((terms_per_doc[doc[i]]/attrib_count) + laplace_factor)

                            tf = p_i * p_i_prev * (1/p_i)
                        else:

                            p_i_next = (bigram_corpus[(doc[i], doc[i+1])]+0.00001)/((terms_per_doc[doc[i]]/attrib_count) + laplace_factor)


                            p_i_prev = (bigram_corpus[(doc[i-1], doc[i])]+0.00001)/((terms_per_doc[doc[i]]/attrib_count) + laplace_factor)

                            tf = p_i * p_i_prev * p_i_next
                        doc_occ = vector_space[doc[i]]
                        idf = math.log(num_docs/(1+doc_occ)) + 1
                        tf_idf = tf * idf
                        vector_index = word_loc_in_headers[doc[i]]
                        vector[count][vector_index] = tf_idf
                    except:
                        continue
                count+=1
                        
            
        
        if method == 'trigram':
            self.cur_method='trigram'
            _,_,_,bigram_corpus = getVectorSpaceBigram(new_text)
            headers, vector_space,terms_per_doc,trigrams_corpus = getVectorSpaceTrigram(new_text)
            
            word_loc_in_headers = {}
            
            if(use_language_vocab == True):
                vector_space = self.vocabulary
            else:
                self.vocabulary = vector_space
            headers = sorted(list(vector_space.keys()))
            for i in range(len(headers)):
                word_loc_in_headers[headers[i]] = i
            TF_IDF_DF = np.asarray([headers])
            count = 0
            attrib_count = len(headers)
            num_elements = len(new_text)
            vector = np.zeros((num_elements, attrib_count))
            num_docs = num_elements
            self.doc_count = num_docs
            for doc in new_text:
                word_ind = 0
                for i in range(len(doc)):
                    try:
                        p_i = terms_per_doc[doc[i]]/attrib_count
                        laplace_factor = 1e-6
                        if(i == 0 and i<len(doc)-2):

                            bigram_score = (bigram_corpus[(doc[i+1], doc[i+2])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_right = (trigrams_corpus[(doc[i], doc[i+1], doc[i+2])])/(p_i+laplace_factor) * bigram_score

                            tf = p_i * p_i_right
                        elif(i == 0 and i<len(doc)-1):
                            p_i_next = 1
                            tf = p_i * (1/p_i) * (1/p_i) * (1/p_i)
                        elif (i == 1 and i<len(doc)-2):

                            bigram_score = (bigram_corpus[(doc[i-1], doc[i])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_middle = (trigrams_corpus[(doc[i-1], doc[i], doc[i+1])])/(p_i+laplace_factor) * bigram_score


                            bigram_score = (bigram_corpus[(doc[i+1], doc[i+2])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_right = (trigrams_corpus[(doc[i], doc[i+1], doc[i+2])])/(p_i+laplace_factor) * bigram_score

                            tf = p_i * p_i_middle *p_i_right * (1/p_i)
                        elif (i == 1 and i<len(doc)-1):

                            bigram_score = (bigram_corpus[(doc[i-1], doc[i])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_middle = (trigrams_corpus[(doc[i-1], doc[i], doc[i+1])])/(p_i+laplace_factor) * bigram_score

                            tf = p_i * p_i_middle * (1/p_i) * (1/p_i)
                        elif(i == len(doc) - 2):

                            bigram_score = (bigram_corpus[(doc[i-2], doc[i-1])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_left = (trigrams_corpus[(doc[i-2], doc[i-1], doc[i])])/(p_i+laplace_factor) * bigram_score


                            bigram_score = (bigram_corpus[(doc[i-1], doc[i])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_middle = (trigrams_corpus[(doc[i-1], doc[i], doc[i+1])])/(p_i+laplace_factor) * bigram_score

                            
                            tf = p_i * p_i_left * p_i_middle * (1/p_i)
                        elif(i == len(doc) - 1):

                            bigram_score = (bigram_corpus[(doc[i-2], doc[i-1])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_left = (trigrams_corpus[(doc[i-2], doc[i-1], doc[i])])/(p_i+laplace_factor) * bigram_score

                            tf = p_i * p_i_left * (1/p_i) * (1/p_i)
                        else:

                            bigram_score = (bigram_corpus[(doc[i-2], doc[i-1])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_left = (trigrams_corpus[(doc[i-2], doc[i-1], doc[i])])/(p_i+laplace_factor) * bigram_score


                            bigram_score = (bigram_corpus[(doc[i-1], doc[i])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_middle = (trigrams_corpus[(doc[i-1], doc[i], doc[i+1])])/(p_i+laplace_factor) * bigram_score


                            bigram_score = (bigram_corpus[(doc[i+1], doc[i+2])])/((terms_per_doc[doc[i]]/attrib_count))
                            p_i_right = (trigrams_corpus[(doc[i], doc[i+1], doc[i+2])])/(p_i+laplace_factor) * bigram_score

                            tf = p_i * p_i_left * p_i_middle * p_i_right
                        doc_occ = vector_space[doc[i]]
                        idf = math.log(num_docs/(1+doc_occ)) + 1
                        tf_idf = tf * idf
                        vector_index = word_loc_in_headers[doc[i]]
                        vector[count][vector_index] = tf_idf
                    except:
                        continue
                count+=1            
        
        
        labels = np.squeeze(np.asarray(raw_data['Label']))
        return headers, vector, labels
    def logistic_loss(self, predicted_label, true_label):
        """
        Optional helper method to code the loss function.
        """
        return [predicted_label-true_label]
    
    def regularizer(self, method=None, lam=None):
        """
        You need to implement at least L1 and L2 regularizer
        """
        if method == 'L1':
            sign_matrix = np.where(self.weights >=0, 1, 0)
            return lam * sign_matrix
        if method == 'L2':
            return lam * self.weights

    def stochastic_gradient_descent(self, data, labels):
        """
        Optional helper method to compute a gradient update for a single point.
        """
        loss_diff = data *loss[0]
        new_weights = self.weights - loss_diff * self.learning_rate
        self.weights = new_weights
        return

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights during stochastic gradient descent.
        """
        self.weights = new_weights

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias during stochastic gradient descent.
        """
        self.bias = new_bias

    def predict_labels(self, data_point, prev_in=None, ex=False):
        """
        Optional helper method to produce predictions for a single data point
        """
        if(ex):
            prod = prev_in
        else:
            prod = np.matmul(data_point, self.weights) + self.bias
        if(prod.any()<0):
            return np.exp(prod)/(1+np.exp(prod))
        return 1/(1+np.exp(-1*prod))

    def train(self, labeled_data, learning_rate=None, max_epochs=None, lam=None, feature_method=None, reg_method=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. 
        This dataframe must have the label of the data points stored in a column called 'Label'. For example, 
        the column labeled_data['Label'] must return the labels of every data point in the dataset. 
        Additionally, this function should not return anything.
        
        'learning_rate' and 'max_epochs' are the same as in HW2. 'reg_method' represents the regularier, 
        which can be 'L1' or 'L2' as in the regularizer function. 'lam' is the coefficient of the regularizer term. 
        'feature_method' can be 'unigram', 'bigram' or 'trigram' as in 'feature_extraction' method. Once you find the optimal 
        values combination, update the default values for all these parameters.

        There is no limitation on how you implement the training process.
        """
        if(learning_rate != None and max_epochs !=None and lam != None):
            self.max_iter = max_epochs
            self.learning_rate = learning_rate
            headers, train_X, train_Y = self.feature_extraction(labeled_data, method=feature_method)
            self.weights = np.zeros((len(headers),))
            self.bias = 0
            for i in range(max_epochs):
                count = 0
                #print("Epoch ", i)
                batch_size = 1
                for j in range(0, len(train_X), batch_size):
                    rows = train_X[j]
                    labels = self.predict_labels(rows)
                    model_result = np.squeeze(labels)
                    #print(rows)
                    expected_result = train_Y[j]
                    loss = [model_result - expected_result]
                    #print("HERE")
                    self.weights = self.weights - self.learning_rate * ((loss[0] * rows) + (self.regularizer(reg_method, lam)))
                    #print("Epoch ", i, " Batch ", j//20)
        else:
            #K - FOLD CROSS VALIDATION
            print("YOU ARE ENTERING K-FOLD VALIDATION BECAUSE YOU DIDN'T PROVIDE HYPERPARAMS")
            headers, train_X, train_Y = self.feature_extraction(labeled_data, method=feature_method)
            data_len = len(train_X)
            block_size = int(data_len/5)
            train_x_blocks = []
            train_y_blocks = []
            for i in range(0, data_len, block_size):
                train_x_blocks.append(train_X[i:i+block_size])
                train_y_blocks.append(train_Y[i:i+block_size])
            lr_values = [0.001, 0.01, 0.05, 0.1]
            max_iter_values = [1, 2, 3, 5, 8, 10]
            lam_values = [0.0001, 0.001, 0.01]
            accuracy_grid = {}
            for i in lr_values:
                accuracy_grid[i] = {}
                for j in max_iter_values:
                    accuracy_grid[i][j] = {}
                    for k in lam_values:
                        accuracy_grid[i][j][k] = []
                        print("MAX_ITER = ", j, " LR = ", i, " LAM = ", k)
                        for l in range(5):
                            validation_x = train_x_blocks[l]
                            validation_y = train_y_blocks[l]
                            new_train = copy.copy(train_x_blocks)
                            new_train_label = copy.copy(train_y_blocks)
                            new_train.pop(l)
                            new_train_label.pop(l)
                            new_train = np.concatenate(new_train, axis=0)
                            new_train_label = np.concatenate(new_train_label, axis=0)
                            weights = cross_valid_train(headers, new_train, new_train_label, i, j, reg_method, k)
                            predictions = np.squeeze(np.matmul(validation_x, weights))
                            #print(predictions)
                            count = 0
                            for m in range(len(predictions)):
                                if((predictions[m]>=0.5 and validation_y[m] == 1) or (predictions[m]<0.5 and validation_y[m] == 0)):
                                    count+=1
                            accuracy_grid[i][j][k].append(count/len(predictions))
                        accuracy_grid[i][j][k] = mean(accuracy_grid[i][j][k])
                        print(weights)
            with open('logistic.json', 'w') as json_file:
                json.dump(accuracy_grid, json_file)
            print(accuracy_grid)
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
        headers, vector, labels = self.feature_extraction(data, use_language_vocab = True, method=self.cur_method)
        predictions = np.squeeze(self.predict_labels(0, np.matmul(vector, self.weights), True))
        
        predicted_labels = list(predictions)
        for i in range(len(predicted_labels)):
            if(predicted_labels[i]<0.5):
                predicted_labels[i] = 0
            else:
                predicted_labels[i] = 1
        return predicted_labels

data = pd.read_csv('data.csv')
train_data = data.sample(frac=0.8)
train_data.reset_index(drop=True, inplace=True)
indexes = list(train_data.index)
test_data = data.drop(indexes)
test_data = test_data.reset_index(drop=True)
logistic_reg = Logistic()
logistic_reg.train(train_data, learning_rate=None, max_epochs=None, feature_method='bigram', reg_method='L1', lam=None)
predictions = logistic_reg.predict(train_data)
count = 0
for i in range(len(predictions)):
    if(predictions[i] == train_data['Label'][i]):
        count+=1
print("Training Accuracy :", count/len(predictions))

predictions = logistic_reg.predict(test_data)
count = 0
for i in range(len(predictions)):
    if(predictions[i] == test_data['Label'][i]):
        count+=1
print("Testing Accuracy :", count/len(predictions))

