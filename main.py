import pandas as p
import math
import copy

def nearestNeighbor(data, features, element): #Returns the class of the nearest neighbor
    nearestDist = float('inf')
    nearestElement = 0
    nearestClass = 0
    for i in range(data.shape[0]):
        if i != element:
            dist = 0
            for f in features:
                dist += (data.loc[i, f] - data.loc[element, f])**2
            dist = math.sqrt(dist)
            if dist < nearestDist:
                nearestDist = dist
                nearestElement = i
                nearestClass = data.loc[i, 0]
    return nearestClass

def forwardsOneOutCV(data, features, newfeature):
    testfeatures = copy.copy(features)
    testfeatures.append(newfeature)
    correct = 0
    for element in range(data.shape[0]):
        if data.loc[element, 0] == nearestNeighbor(data, testfeatures, element):
            correct += 1
    return correct/data.shape[0]




def forwardSelection(data):
    chosen = []
    notchosen = []
    notchosen.extend(range(1, data.shape[1]))
    finalacc = 0
    bestfeatureset = []
    for level in range(1, data.shape[1]):
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
        if bestacc > finalacc:
            finalacc = bestacc
            bestfeatureset = copy.copy(chosen)
    return finalacc, bestfeatureset


filename = 'SueDataSmall96.txt'
data = p.read_csv(filename, sep="  ", engine='python', header=None)
print(data)
x, y = forwardSelection(data)
feats = ' '.join([str(i) for i  in y])
print('Best feature set is: [' + feats + '] with an accuracy of ' + str(x))