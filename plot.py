import matplotlib.pyplot as plt
import json
import math
with open('perceptron.json', 'r') as file:
    perceptron_data = json.load(file)
with open('logistic.json', 'r') as file:
    logistic_data = json.load(file)

learning_rates = list(perceptron_data.keys())
max_iters = list(perceptron_data[learning_rates[0]].keys())
print(learning_rates)
print(max_iters)

for i in max_iters:
    x_val = []
    y_val = []
    for j in learning_rates:
        x_val.append(math.log(eval(j)))
        y_val.append(perceptron_data[j][i])
    plt.plot(x_val, y_val)
    plt.xlabel('Log Learning Rates')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Perceptron Learning Rate vs Accuracy at Iters ="+ str(i))
    plt.show()
    
    
for i in max_iters:
    x_val = []
    y_val = []
    for j in learning_rates:
        x_val.append(eval(j))
        y_val.append(logistic_data[j][i])
    plt.plot(x_val, y_val)
    plt.xlabel('Log Learning Rates')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Logistic Learning Rate vs Accuracy at Iters = "+ str(i))
    plt.show()
    
    
