import pandas as p #For reading in data
import math #sqrt()
import copy #For shallow copying feature sets
import time #For finding runtime

def nearestNeighbor(data, features, element): #Returns the class of the nearest neighbor
    nearestDist = float('inf')
    nearestElement = 0
    nearestClass = 0
    for i in range(len(data)):
        if i != element:
            dist = 0
            for f in features:
                dist += (data[element][f] - data[i][f])**2
            dist = math.sqrt(dist)
            if dist < nearestDist:
                nearestDist = dist
                nearestElement = i
                nearestClass = data[i][0]
    #print('Nearest element to ' + str(element) + ' is ' + str(nearestElement) + ' with a class of ' + str(nearestClass) + ' and a distance of ' + str(nearestDist))
    return nearestClass

def forwardsOneOutCV(data, features, newfeature):#Makes combined feature set and runs nearestNeighbor on all elements using those features
    testfeatures = copy.copy(features)
    testfeatures.append(newfeature)
    correct = 0
    for element in range(len(data)):
        if data[element][0] == nearestNeighbor(data, testfeatures, element):
            correct += 1
    return correct/len(data)


def forwardSelection(data):#Wrapper for forward selection
    chosen = []
    notchosen = []
    notchosen.extend(range(1, len(data[0])))
    finalacc = 0
    bestfeatureset = []
    for level in range(1, len(data[0])):
        print('Testing features for level ' + str(level))
        bestacc = 0
        bestfeat = 0
        for feat in notchosen:
            print('\tTesting feature ' + str(feat))
            acc = forwardsOneOutCV(data, chosen, feat)
            print('\t\tAccuracy: ' + str(acc))
            if acc > bestacc:
                bestacc = acc
                bestfeat = feat
        print('Best feature for this level is ' + str(bestfeat) + ', adding it to the feature set')
        chosen.append(bestfeat)
        notchosen.remove(bestfeat)
        print(chosen)
        if bestacc > finalacc:
            finalacc = bestacc
            bestfeatureset = copy.copy(chosen)
    return finalacc, bestfeatureset

def backwardsOneOutCV(data, features, removefeature):#Makes split feature set and runs nearestNeighbor on all elements using those features
    testfeatures = copy.copy(features)
    testfeatures.remove(removefeature)
    correct = 0
    for element in range(len(data)):
        if data[element][0] == nearestNeighbor(data, testfeatures, element):
            correct += 1
    return correct/len(data)


def backwardElimination(data):#Wrapper for backwards elimination
    chosen = []
    notchosen = []
    chosen.extend(range(1, len(data[0])))
    finalacc = 0
    bestfeatureset = []
    for level in range(1, len(data[0])):
        print('Testing features for level ' + str(level))
        bestacc = 0
        bestfeat = 0
        for feat in chosen:
            print('\tTesting without feature ' + str(feat))
            acc = backwardsOneOutCV(data, chosen, feat)
            print('\t\tAccuracy: ' + str(acc))
            if acc > bestacc:
                bestacc = acc
                bestfeat = feat
        print('Worst feature for this level is ' + str(bestfeat) + ', removing it from the feature set')
        notchosen.append(bestfeat)
        chosen.remove(bestfeat)
        print(chosen)
        if bestacc > finalacc:
            finalacc = bestacc
            bestfeatureset = copy.copy(chosen)
    return finalacc, bestfeatureset

filename = 'SueDataSmall96.txt'
data = p.read_csv(filename, sep="  ", engine='python', header=None)#Read in file
print(data)
data = data.values.tolist()#Convert dataframe to list for better runtime
start = time.time()#Start timer
x, y = backwardElimination(data)
feats = ', '.join([str(i) for i  in y])
print('Best feature set is: [' + feats + '] with an accuracy of ' + str(x))
#print(time.time()-start)#Print runtime