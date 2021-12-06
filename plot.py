import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import json
import math

with open('logistic.json', 'r') as file:
    logistic_data = json.load(file)

learning_rates = list(logistic_data.keys())
max_iters = list(logistic_data[learning_rates[0]].keys())
lam_vals = list(logistic_data[learning_rates[0]][max_iters[0]].keys())
print(learning_rates)
print(max_iters)
print(lam_vals)

ensemble_acc = {1:0.797, 3:0.844, 5:0.894, 7:0.891, 9:0.897}
plt.plot(list(ensemble_acc.keys()), list(ensemble_acc.values()))
plt.xlabel('Number of Models')
plt.ylabel('Accuracy')
plt.legend()
plt.title('N_Clf vs Accuracy For bagging')
plt.show()


ind = 1
for i in max_iters:
    x_val = []
    y_val = []
    for j in learning_rates:
        x_val.append(eval(j))
        y_val.append(logistic_data[j][i]['0.01'])
    
    plt.subplot(2, 3, ind)
    ind +=1
    plt.plot(x_val, y_val)
    if(ind > 4):
        plt.xlabel('Learning Rates')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("LR vs Accuracy at Iters = "+ str(i))
plt.show()
    
ind = 1
for i in learning_rates:
    x_val = []
    y_val = []
    for j in max_iters:
        x_val.append(eval(j))
        y_val.append(logistic_data[i][j]['0.01'])

    plt.subplot(2, 3, ind)
    plt.plot(x_val, y_val)
    ind+=1
    if(ind >4):
        plt.xlabel('Max Iters')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Max Iters vs Accuracy at LR = "+ str(i))
plt.show()
x_val = []
y_val = []
for i in max_iters:

    x_val.append(eval(i))
    y_val.append(logistic_data['0.001'][i]['0.01'])
plt.plot(x_val, y_val, label='validation_accuracy')
plt.plot(x_val, [0.8897, 0.91, 0.93, 0.94, 0.955, 0.97], label='Train Accuracy')
plt.xlabel('Max Iters')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Max Iters vs Accuracy at LR = 0.001, LAM=0.01')
plt.show()
x_val = []
y_val = []
for i in learning_rates:

    x_val.append(eval(i))
    y_val.append(logistic_data[i]['3']['0.01'])
plt.plot(x_val, y_val, label='validation_accuracy')
plt.plot(x_val, [0.92, 0.87, 0.83, 0.79], label='Train Accuracy')
plt.xlabel('Learning Rates')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Rates vs Accuracy at Epoch = 3, LAM=0.01')
plt.show()
x_val = []
y_val = []
for i in lam_vals:

    x_val.append(eval(i))
    y_val.append(logistic_data['0.001']['3'][i])
plt.plot(x_val, y_val, label='validation_accuracy')
plt.plot(x_val, [0.92, 0.83, 0.77, 0.64, 0.59], label='Train Accuracy')
plt.xlabel('Regularization Param')
plt.ylabel('Accuracy')
plt.legend()
plt.title('LAM vs Accuracy at Epoch = 3, LR=0.001')
plt.show()

with open('reg_feat.json', 'r') as file:
    reg_feat = json.load(file)

L1_train = [reg_feat['unigram']['L1'][0], reg_feat['bigram']['L1'][0],reg_feat['trigram']['L1'][0]]
L1_test =  [reg_feat['unigram']['L1'][1], reg_feat['bigram']['L1'][1],reg_feat['trigram']['L1'][1]]
L1_unseen = [reg_feat['unigram']['L1'][2], reg_feat['bigram']['L1'][2],reg_feat['trigram']['L1'][2]]

L2_train = [reg_feat['unigram']['L2'][0], reg_feat['bigram']['L2'][0],reg_feat['trigram']['L2'][0]]
L2_test = [reg_feat['unigram']['L2'][1], reg_feat['bigram']['L2'][1],reg_feat['trigram']['L2'][1]]
L2_unseen = [reg_feat['unigram']['L2'][2], reg_feat['bigram']['L2'][2],reg_feat['trigram']['L2'][2]]

plt.plot(['unigram', 'bigram', 'trigram'], L1_train, label='Train Acc L1')
plt.plot(['unigram', 'bigram', 'trigram'], L1_test, label='Test Acc L1')

plt.plot(['unigram', 'bigram', 'trigram'], L1_unseen, label='UnseenData Acc L1')

plt.plot(['unigram', 'bigram', 'trigram'], L2_train, label='Train Acc L2')

plt.plot(['unigram', 'bigram', 'trigram'], L2_test, label='Test Acc L2')

plt.plot(['unigram', 'bigram', 'trigram'], L2_unseen, label='UnseenData Acc L2')

plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title("Method, Reg_method vs Accuracy")
plt.legend()
plt.show()

