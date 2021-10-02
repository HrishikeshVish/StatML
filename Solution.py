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
                
        def Entropy(self, row):
                distinct_class = set(row)
                entropy = 0
                net = len(row)
                for cl in distinct_class:
                        frac = (row.count(cl)/net)
                        entropy -= frac * (math.log(frac)/math.log(2.0))
                return entropy


        def informationGain(self, res, attrib):
                attrib_class = set(attrib)
                classDict = {}
                for i in attrib_class:
                        classDict[i] = []
                for j in range(len(attrib)):
                        classDict[attrib[j]].append(res[j])
                entSum = 0

                for i in attrib_class:

                        entSum += (len(classDict[i])/len(res)) * self.Entropy(classDict[i])
                return self.Entropy(res) - entSum
        
        def getCols(self, dataSet):
                colLen = len(dataSet[0])
                cols = []
                for i in range(colLen):
                        col = []
                        for j in dataSet:
                        
                                col.append(j[i])
                        col.pop(0)
                        cols.append(col)       
                return cols
                
        def Split(self, attribNumber, attribCol, dataSet, header):
                attrib_class = set(attribCol)
                newData = []
                for i in attrib_class:
                        splitPart = []
                        newHeader = copy.copy(header)
                        newHeader.pop(attribNumber)
                        splitPart.append(newHeader)
                        for j in range(1, len(dataSet)):

                                if(dataSet[j][attribNumber] == i):
                                        temp = copy.copy(dataSet[j])
                                        temp.pop(attribNumber)
                                        splitPart.append(temp)
                                
                        newData.append(splitPart)

                return dataSet[0][attribNumber], newData

        def buildLayer(self, train, remCols, curHeader):
                columns = self.getCols(train)
                infoGain = []
                corCol = []
                if(len(train) == 3 and train[1][0] == train[2][0]):
                        train[1][1] = train[2][1]
                for i in range(len(columns)-1):

                        if(type(columns[i][0]) == int or type(columns[i][0]) == float or columns[i][0] =='?'):
                                infoGain.append(-1990)
                                corCol.append(i)
                        #elif(type(columns[i][0]) == tuple):
                         #       infoGain.append(1.6)
                          #      corCol.append(i)
                        else:

                                infoGain.append(self.informationGain(columns[len(columns)-1], columns[i]))
                                corCol.append(i)

                maxGain = max(infoGain)

                
                index = infoGain.index(maxGain)
                colIndex = corCol[index]

                attribNumber, newData = self.Split(colIndex, columns[colIndex], train, curHeader)
                return list(set(columns[colIndex])), attribNumber, newData
        
        def buildTree(self, trainDataSet, cur):
                decisionTree = {}
                res = ['+','-']
                ind = random.randint(0,1)
                tree = treeNode(0, attribs=None, children={}, value=res[0], cur=cur)
                if(len(trainDataSet) == 1):
                        return tree
                if(len(trainDataSet[0]) == 1):
                        return tree
                childCols, attribNumber, newData = self.buildLayer(trainDataSet, [], trainDataSet[0])
                tree = treeNode(attribNumber, attribs=childCols, children={}, value=None, cur=cur)
                for i in childCols:
                        if i not in decisionTree:
                                decisionTree[i] = {}
                        if i not in tree.childNodes:
                                tree.childNodes[i] = {}
                count = 0

                for i in range(len(newData)):

                        cols = self.getCols(newData[i])

                        if(len(set(cols[len(cols)-1])) == 1):
                                

                                childNode = treeNode(attribNumber, value=cols[len(cols)-1][0], cur=childCols[i])
                                tree.childNodes[childCols[i]] = childNode
                        else:
                                

                        
                                tree.childNodes[childCols[i]] = self.buildTree(newData[i], childCols[i])

                        count = count + 1
        

                return tree
        def buildTreeDepth(self, trainDataSet, cur, maxDepth, curDepth):
                decisionTree = {}
                res = ['+','-']
                ind = random.randint(0,1)
                tree = treeNode(0, attribs=None, children={}, value=res[0], cur=cur)
                if(len(trainDataSet) == 1):
                        return tree
                childCols, attribNumber, newData = self.buildLayer(trainDataSet, [], trainDataSet[0])
        
        
                tree = treeNode(attribNumber, attribs=childCols, children={}, value=None, cur=cur)
                for i in childCols:
                        if i not in decisionTree:
                                decisionTree[i] = {}
                        if i not in tree.childNodes:
                                tree.childNodes[i] = {}
                count = 0
                if(curDepth == maxDepth):
                        for i in range(len(newData)):
                                cols = self.getCols(newData[i])
                                try:
                                        output = mode(cols[len(cols)-1])
                                except:
                                        output = res[0]
                                childNode = treeNode(attribNumber, value = output, cur=childCols[i])
                                tree.childNodes[childCols[i]] = childNode
                        return tree
                        
                for i in range(len(newData)):

                        cols = self.getCols(newData[i])

                        if(len(set(cols[len(cols)-1])) == 1):
                                

                                childNode = treeNode(attribNumber, value=cols[len(cols)-1][0], cur=childCols[i])
                                tree.childNodes[childCols[i]] = childNode
                        else:
                                

                        
                                tree.childNodes[childCols[i]] = self.buildTreeDepth(newData[i], childCols[i], maxDepth, curDepth+1)

                        count = count + 1
        

                return tree

        



        


def preProcessing(trainDataSet,columns):
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

        return trainDataSet



def numToCat(trainDataSet, columns):
        resClass = columns[len(columns)-1]
        resClass = [1 if x=='+' else 0 for x in resClass]

        categorical = {}
        for i in range(len(columns)-1):
                if(type(columns[i][0])!=str):
                        while('?' in columns[i]):
                                qInd = columns[i].index('?')

                                temp = copy.copy(columns[i])

                                temp.pop(qInd)

                                while('?' in temp):
                                        temp.remove('?')
                                columns[i][qInd] = mean(temp)
                        
                        zipped = zip(columns[i],resClass)
                        sortedpairs = sorted(dict(sorted(zipped, key=lambda x: x[1], reverse= True)).items(), key=lambda x: x[0])
                        categorical[i] = []
                        temp = copy.copy(columns[i])
                        sorted(temp)
                        cats = []
                        curLabel = sortedpairs[0][1]
                        cats.append(sortedpairs[0][0])
                        for k in range(1,len(sortedpairs)):
                                if(sortedpairs[k][1] !=curLabel):
                                        cats.append(sortedpairs[k][0])
                                        curLabel = sortedpairs[k][1]
                        cur = 0.0

                        categorical[i] = cats

                        

        cols = [1,2,7,10,13,14]
        for i in range(1, len(trainDataSet)):
                for j in cols:
                        value = trainDataSet[i][j]
                        k = 0
                        
                        while(k<len(categorical[j])-1 and value > categorical[j][k]):
                                k+=1
                        if(k==0):
                                trainDataSet[i][j] = (0.0, categorical[j][1])
                        else:
                                
                                trainDataSet[i][j] = (categorical[j][k-1], categorical[j][k])
        return trainDataSet, categorical



def printTreeRecurse(tree, indent):
        
        if(tree.val !=None):
                print(indent*'-', indent, tree.cur, tree.attribNumber, end = '')
                print(tree.val)
        else:
                print(indent*'-', indent, tree.cur, tree.attribNumber, tree.attributes, end = '')
                print()
                for i in tree.attributes:
                        printTreeRecurse(tree.childNodes[i], indent+1)

def Vote(iterTree):
        if(iterTree.val !=None):
                return iterTree.val
        outs = []
        for i in iterTree.attributes:
                outs.append(Vote(iterTree.childNodes[i]))
        try:
                response = mode(outs)
        except:
                response = '+'
        return response
                

def getNextNode(testRow, curRow, iterTree, iterTreevals):
        nextNodes = iterTree.attributes
        childNode = treeNode(iterTree.attribNumber, value=iterTreevals[curRow-1])
        
        for i in nextNodes:
                tempTree = copy.copy(iterTree)
                tempTree = tempTree.childNodes[i]
                curCol = int(iterTree.attribNumber[1:])-1
                
                if(tempTree.val!=None):
                        childNode.attribNumber = tempTree.attribNumber
                        childNode.attributes = tempTree.attributes
                        childNode.cur = tempTree.cur
                        return childNode
                else:
                        responses = Vote(iterTree)
                        childNode.val = responses
        return childNode

def DecisionTreeBounded(maxDepth = 15):
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
	newvals = []
	mainObj = treeNode(0)
	trainDataSet = [['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']]
	TrainFile = open('train.txt', 'r')
	for line in TrainFile:
                inputLine = line.strip().split('\t')
                for x in range(len(inputLine)):
                        if inputLine[x].isdigit() or '.' in inputLine[x]:
                                inputLine[x] = eval(inputLine[x])
                trainDataSet.append(inputLine)
                                
	
	columns = mainObj.getCols(trainDataSet)
	trainDataSet = preProcessing(trainDataSet, columns)
	trainDataSet, categories = numToCat(trainDataSet, columns)

	labels = []
	cols_train = mainObj.getCols(trainDataSet)
	#print(maxDepth)
	tree2 = mainObj.buildTreeDepth(trainDataSet, 'root', maxDepth,1)
	#printTreeRecurse(tree2, 0)
	labels.append(cols_train[15])
	
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

                                iterTree = iterTree.childNodes[trainDataSet[i][curCol]]
                                curTreeVal = iterTree.val

                newvals.append(curTreeVal)
                if(curTreeVal == trainDataSet[i][len(trainDataSet[i])-1]):
                        F1_train +=1
        
	F1_test = 0
	testDataSet = [['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']]
	TestFile = open('test.txt', 'r')
	for line in TestFile:
                inputLine = line.strip().split('\t')
                for x in range(len(inputLine)):
                        if inputLine[x].isdigit() or '.' in inputLine[x]:
                                inputLine[x] = eval(inputLine[x])
                                value = inputLine[x]
                                k = 0
                                while(k<(len(categories[x])-1) and value > categories[x][k]):
                                        k+=1
                                if(k==0):
                                        inputLine[x] = (0.0, categories[x][1])
                                else:
                                        inputLine[x] = (categories[x][k-1], categories[x][k])
                testDataSet.append(inputLine)            
                                        
                                   
	labels.append(newvals)
	cols_test = mainObj.getCols(testDataSet)
	iterTreevals = cols_test[15]
	newvals = []
	labels.append(cols_test[15])
	for i in range(1, len(testDataSet)):
        
                iterTree = copy.copy(tree2)
                curRow = testDataSet[i]
                curTreeVal = iterTree.val
                rowLen = 15
                res = ['+', '-']

                while(curTreeVal == None):
                        curCol = int(iterTree.attribNumber[1:])-1
                        
                        if(iterTree.val !=None):
                                curTreeVal = iterTree.val
                        else:

                                try:
                                        iterTree = iterTree.childNodes[testDataSet[i][curCol]]
                                        curTreeVal = iterTree.val
                                except:

                                        index = random.randint(0, len(iterTree.attributes)-1)

                                        iterTree = getNextNode(testDataSet[i],i, iterTree, iterTreevals)
                                        
                                        curTreeVal = iterTree.val

                newvals.append(curTreeVal)        

	labels.append(newvals)

	return labels





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
	#tree2 = buildTreeDepth(trainDataSet, 'root', 7,1)
	return DecisionTreeBounded(15)


