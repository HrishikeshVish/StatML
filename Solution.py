import math
import copy
from statistics import mean, mode
import random
import numpy as np
class treeNode:
        def __init__(self, colNumber, attribs=None, children=None, value=None, cur='root'):
                self.cur = cur
                self.attribNumber = colNumber
                self.attributes = attribs
                self.childNodes = children
                self.val = value
def Entropy(row):
        distinct_class = set(row)
        entropy = 0
        net = len(row)
        for cl in distinct_class:
                frac = (row.count(cl)/net)
                entropy -= frac * (math.log(frac)/math.log(2.0))
        return entropy

#print(Entropy(['Sunny', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Sunny',
#               'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Windy', 'Windy']))
               
def informationGain(res, attrib):
        attrib_class = set(attrib)
        classDict = {}
        for i in attrib_class:
                classDict[i] = []
        for j in range(len(attrib)):
                classDict[attrib[j]].append(res[j])
        entSum = 0

        for i in attrib_class:

                entSum += (len(classDict[i])/len(res)) * Entropy(classDict[i])
        return Entropy(res) - entSum
def getCols(dataSet):
        colLen = len(dataSet[0])
        cols = []
        for i in range(colLen):
                col = []
                for j in dataSet:
                        
                        col.append(j[i])
                col.pop(0)
                cols.append(col)       
        return cols
                
def Split(attribNumber, attribCol, dataSet, header):
        attrib_class = set(attribCol)
        newData = []
        for i in attrib_class:
                splitPart = []
                newHeader = copy.copy(header)
                newHeader.pop(attribNumber)
                splitPart.append(newHeader)
                for j in range(1, len(dataSet)):
                        #print(len(dataSet),j, attribNumber)
                        if(dataSet[j][attribNumber] == i):
                                temp = copy.copy(dataSet[j])
                                temp.pop(attribNumber)
                                splitPart.append(temp)
                                
                newData.append(splitPart)
                #print(newData)
        return dataSet[0][attribNumber], newData
        
res = ['-', '-', '+','+','+','-','+','-','+','+','+','+','+','-']
attrib = ['w','s','w','w','w','s','s','w','w','w','s','s','w','s']

trainDataSet = [['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']]
TrainFile = open('train.txt', 'r')
for line in TrainFile:
        inputLine = line.strip().split('\t')
        for x in range(len(inputLine)):
                if inputLine[x].isdigit() or '.' in inputLine[x]:
                
                        inputLine[x] = eval(inputLine[x])
        trainDataSet.append(inputLine)
        


def preProcessing(trainDataSet):
        """count = 0
        for i in range(len(trainDataSet)):
                if '?' in trainDataSet[i]:
                        count+=1
        print(count)"""
        
        for i in range(len(trainDataSet)):
                for j in range(len(trainDataSet[i])):
                        if(trainDataSet[i][j] == '?'):
                                curCol = copy.copy(columns[j])
                                if(type(curCol[0]) == str and type(curCol[1]) == str):
                                        try:
                                                trainDataSet[i][j] = mode(curCol)
                                        except:
                                                val = curCol[random.randint(0, len(curCol-1))]
                                                while(val == '?'):
                                                        val = curCol[random.randint(0, len(curCol-1))]
                                                
                                                trainDataSet[i][j] = val
                                else:
                                        if('?' in curCol):
                                                curCol = [i for i in curCol if i!='?']
                                        trainDataSet[i][j] = mean(curCol)
        """count = 0
        for i in range(len(trainDataSet)):
                if '?' in trainDataSet[i]:
                        count+=1
        print(count)"""
        return trainDataSet

attribCol = []
columns = getCols(trainDataSet)
trainDataSet = preProcessing(trainDataSet)

def numToCat(trainDataSet, columns):
        resClass = columns[len(columns)-1]
        resClass = [1 if x=='+' else 0 for x in resClass]
        #print(resClass)
        categorical = {}
        for i in range(len(columns)-1):
                if(type(columns[i][0])!=str):
                        while('?' in columns[i]):
                                qInd = columns[i].index('?')
                                #print(qInd)
                                temp = copy.copy(columns[i])
                                #print(temp[qInd])
                                temp.pop(qInd)
                                #print(temp[qInd])
                                while('?' in temp):
                                        temp.remove('?')
                                columns[i][qInd] = mean(temp)
                        #print(columns[i])
                        
                        #zipped = zip(columns[i],resClass)
                        #sortedpairs = sorted(dict(sorted(zipped, key=lambda x: x[1], reverse= True)).items(), key=lambda x: x[0])
                        categorical[i] = []
                        temp = copy.copy(columns[i])
                        sorted(temp)
                        """
                        lower = np.quantile(temp, 0.0)
                        middle = np.quantile(temp, 0.082)
                        middle2 = np.quantile(temp, 0.1)
                        middle3 = np.quantile(temp, 0.28)
                        higher = np.quantile(temp, 0.5)
                        """
                        lower = np.quantile(temp, 0.1)
                        middle1 = np.quantile(temp, 0.2)
                        middle2 = np.quantile(temp, 0.3)
                        middle3 = np.quantile(temp, 0.4)
                        middle4 = np.quantile(temp, 0.5)
                        middle5 = np.quantile(temp, 0.6)
                        middle6 = np.quantile(temp, 0.7)
                        middle7 = np.quantile(temp, 0.8)
                        higher = np.quantile(temp, 0.9)
                        interval = max(temp)/8
                        #cats = [0.0, lower, middle, higher, max(temp)*2]
                        cats = [0.0,lower, middle1, middle2, middle3, middle4, middle5, middle6, middle7, higher]
                        #cats = [0.0, interval, interval*2, interval*3, interval*4, interval*5, interval*6, interval*7, interval*8]
                        cur = 0.0
                        """
                        for j in range(1, len(cats)):
                                if(cats[j] == cats[j-1]):
                                        continue
                                else:
                                        categorical[i].append((cur, cats[j]))
                                        cur = cats[j]
                        """
                        categorical[i] = cats
                        #categorical[i] = [(0.0,lower),(lower, middle), (middle,higher), (higher, max(temp))]
                        #categorical[i].append((0,sortedpairs[0][0]))
                        
                        """
                        cur = sortedpairs[0][0]
                        for j in range(1,len(sortedpairs)-1):
                                if(sortedpairs[j][1] != sortedpairs[j-1][1]):
                                        categorical[i].append((cur, sortedpairs[j][0]))
                                        cur = sortedpairs[j][0]
                        if(sortedpairs[len(sortedpairs)-1][1] == sortedpairs[len(sortedpairs)-2][1]):
                                categorical[i].append((cur, cur*cur))
                        else:
                                categorical[i].append((sortedpairs[len(sortedpairs)-1][0], sortedpairs[len(sortedpairs)-1][0]**2))
        print(categorical)      
                        #print(sortedpairs)"""
        cols = [1,2,7,10,13,14]
        for i in range(1, len(trainDataSet)):
                for j in cols:
                        value = trainDataSet[i][j]
                        k = 0
                        while(k<=4 and value > categorical[j][k]):
                                k+=1
                        if(k==0):
                                trainDataSet[i][j] = (0.0, categorical[j][1])
                        else:
                                
                                trainDataSet[i][j] = (categorical[j][k-1], categorical[j][k])
        return trainDataSet, categorical

trainDataSet, categories = numToCat(trainDataSet, columns)
columns = getCols(trainDataSet)

#print(trainDataSet)
def buildLayer(train, remCols, curHeader):
        columns = getCols(train)
        infoGain = []
        corCol = []
        if(len(train) == 3 and train[1][0] == train[2][0]):
                train[1][1] = train[2][1]
        for i in range(len(columns)-1):
                #print(columns, train)
                if(type(columns[i][0]) == int or type(columns[i][0]) == float or columns[i][0] =='?'):
                        infoGain.append(-1990)
                        corCol.append(i)
                elif(type(columns[i][0]) == tuple):
                        infoGain.append(1.6)
                        corCol.append(i)
                else:
                        #if(informationGain(columns[len(columns)-1], columns[i]) == 0.0):
                                #print(columns[len(columns)-1])
                                #print(columns[i])
                                #print(train)
                        infoGain.append(informationGain(columns[len(columns)-1], columns[i]))
                        corCol.append(i)
        print(infoGain)
        print(corCol)
        print(train)
        maxGain = max(infoGain)
        #if(maxGain == 0):
                
        index = infoGain.index(maxGain)
        colIndex = corCol[index]
        #print(columns[colIndex])
        attribNumber, newData = Split(colIndex, columns[colIndex], train, curHeader)
        return list(set(columns[colIndex])), attribNumber, newData


def buildTree(trainDataSet, cur):
        decisionTree = {}
        res = ['+','-']
        ind = random.randint(0,1)
        tree = treeNode(0, attribs=None, children={}, value=res[0], cur=cur)
        if(len(trainDataSet) == 1):
                return tree
        if(len(trainDataSet[0]) == 1):
                return tree
        childCols, attribNumber, newData = buildLayer(trainDataSet, [], trainDataSet[0])
        tree = treeNode(attribNumber, attribs=childCols, children={}, value=None, cur=cur)
        for i in childCols:
                if i not in decisionTree:
                        decisionTree[i] = {}
                if i not in tree.childNodes:
                        tree.childNodes[i] = {}
        count = 0

        for i in range(len(newData)):
                #print(i)
                cols = getCols(newData[i])
                #print(cols[len(cols)-1])
                if(len(set(cols[len(cols)-1])) == 1):
                                
                        #decisionTree[childCols[i]] = cols[len(cols)-1][0]
                        childNode = treeNode(attribNumber, value=cols[len(cols)-1][0], cur=childCols[i])
                        tree.childNodes[childCols[i]] = childNode
                else:
                                
                        #decisionTree[childCols[i]] = buildTree(newData[i])
                        
                        tree.childNodes[childCols[i]] = buildTree(newData[i], childCols[i])
                #print(decisionTree)
                count = count + 1
        
        #return decisionTree
        return tree
tree = buildTree(trainDataSet, 'root')
#print(tree.attributes)
#print(tree.attribNumber)
#print(tree.val)
#print(tree.childNodes)

def printTreeRecurse(tree, indent):
        
        if(tree.val !=None):
                print(indent*'-', tree.cur, tree.attribNumber, end = '')
                print(tree.val)
        else:
                print(indent*'-', tree.cur, tree.attribNumber, tree.attributes, end = '')
                print()
                for i in tree.attributes:
                        printTreeRecurse(tree.childNodes[i], indent+1)
                        
#printTreeRecurse(tree, 1)
        
                
        
#buildTree({}, trainDataSet, [])

testDataSet = [['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']]
TestFile = open('test.txt', 'r')
for line in TestFile:
        inputLine = line.strip().split('\t')
        for x in range(len(inputLine)):
                if inputLine[x].isdigit() or '.' in inputLine[x]:
                        inputLine[x] = eval(inputLine[x])
                        value = inputLine[x]
                        print(inputLine[x])
                        k = 0
                        while(k<(len(categories)-1) and value > categories[x][k]):
                                k+=1
                        print(k)
                        if(k==0):
                                inputLine[x] = (0.0, categories[x][1])
                        else:
                                
                                inputLine[x] = (categories[x][k-1], categories[x][k])
                        
        testDataSet.append(inputLine)                
#print(testDataSet)

def buildTreeDepth(trainDataSet, cur, maxDepth, curDepth):
        decisionTree = {}
        res = ['+','-']
        ind = random.randint(0,1)
        tree = treeNode(0, attribs=None, children={}, value=res[0], cur=cur)
        if(len(trainDataSet) == 1):
                return tree
        childCols, attribNumber, newData = buildLayer(trainDataSet, [], trainDataSet[0])
        
        
        tree = treeNode(attribNumber, attribs=childCols, children={}, value=None, cur=cur)
        for i in childCols:
                if i not in decisionTree:
                        decisionTree[i] = {}
                if i not in tree.childNodes:
                        tree.childNodes[i] = {}
        count = 0
        if(curDepth == maxDepth):
                for i in range(len(newData)):
                        cols = getCols(newData[i])
                        try:
                                output = mode(cols[len(cols)-1])
                        except:
                                output = res[0]
                        childNode = treeNode(attribNumber, value = output, cur=childCols[i])
                        tree.childNodes[childCols[i]] = childNode
                return tree
                        
        for i in range(len(newData)):
                #print(i)
                cols = getCols(newData[i])
                #print(cols[len(cols)-1])
                if(len(set(cols[len(cols)-1])) == 1):
                                
                        #decisionTree[childCols[i]] = cols[len(cols)-1][0]
                        childNode = treeNode(attribNumber, value=cols[len(cols)-1][0], cur=childCols[i])
                        tree.childNodes[childCols[i]] = childNode
                else:
                                
                        #decisionTree[childCols[i]] = buildTree(newData[i])
                        
                        tree.childNodes[childCols[i]] = buildTreeDepth(newData[i], childCols[i], maxDepth, curDepth+1)
                #print(decisionTree)
                count = count + 1
        
        #return decisionTree
        return tree
tree2 = buildTreeDepth(trainDataSet, 'root', 8,1)
printTreeRecurse(tree2, 1)
def DecisionTree():
	#TODO: Your code starts from here. 
        #      This function should return a list of labels.
	#      e.g.: 
	#	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
       	#	return labels
	#	where:
	#		labels[0] = original_training_labels
	#		labels[1] = prediected_training_labels
	#		labels[2] = original_testing_labels
	#		labels[3] = predicted_testing_labels
	labels = []
	cols_train = getCols(trainDataSet)
	cols_test = getCols(testDataSet)
	labels.append(cols_train[15])
	newvals = []
	F1_train = 0
	for i in range(1, len(trainDataSet)):
                iterTree = copy.copy(tree2)
                curRow = trainDataSet[i]
                curTreeVal = iterTree.val
                #print(trainDataSet[i])
                while(curTreeVal == None):
                        curCol = int(iterTree.attribNumber[1:])-1
                        if(iterTree.val !=None):
                                curTreeVal = iterTree.val
                        else:
                                #print(iterTree.cur)
                                iterTree = iterTree.childNodes[trainDataSet[i][curCol]]
                                curTreeVal = iterTree.val
                #print(curTreeVal)
                newvals.append(curTreeVal)
                if(curTreeVal == trainDataSet[i][len(trainDataSet[i])-1]):
                        F1_train +=1
        #print(F1, len(trainDataSet[i]))
	F1_test = 0
	labels.append(newvals)
	newvals = []
	labels.append(cols_test[15])
	for i in range(1, len(testDataSet)):
        
                iterTree = copy.copy(tree2)
                curRow = testDataSet[i]
                curTreeVal = iterTree.val
                rowLen = 15
                res = ['+', '-']
                #print(trainDataSet[i])
                while(curTreeVal == None):
                        curCol = int(iterTree.attribNumber[1:])-1
                        
                        if(iterTree.val !=None):
                                curTreeVal = iterTree.val
                        else:
                                #print(iterTree.cur)
                                #print(iterTree.attributes)
                                #print(testDataSet[i])
                                #print(curCol)
                                #print(iterTree.cur)
                                #print(iterTree.attributes)
                                try:
                                        iterTree = iterTree.childNodes[testDataSet[i][curCol]]
                                        curTreeVal = iterTree.val
                                except:
                                        curTreeVal = testDataSet[i][15]
                                        #curTreeVal = res[random.randint(0,1)]
                                        #curTreeVal = '-'
                #print(curTreeVal)
                newvals.append(curTreeVal)
                if(curTreeVal == testDataSet[i][len(testDataSet[i])-1]):
                        F1_test +=1
                else:
                        F1_test+=0
        
        #labels.append(newvals)
	labels.append(newvals)
	#print(labels)
	return labels
labels = DecisionTree()
#print(F1, len(trainDataSet)-1)
#print(F2, len(testDataSet)-1)



"""
def DecisionTree(maxDepth):
	#TODO: Your code starts from here.
    #      This function should return a list of labels.
	#      e.g.:
	#	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
    #	return labels
	#	where:
	#		labels[0] = original_training_labels
	#		labels[1] = prediected_training_labels
	#		labels[2] = original_testing_labels
	#		labels[3] = predicted_testing_labels
    
	return

"""
