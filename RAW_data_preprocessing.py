import json
import random


with open('./data/true-false-qa-RAW', encoding='UTF-8') as f:
    data = json.load(f)

nbData = len(data)
nbTrain = 1944
nbTest = 216


random.shuffle(data)
for i in range(nbData):
    data[i]['id'] = i

trainData = data[:nbTrain]
testData = data[nbTrain+1:]

with open('./data/true-false-qa-TRAIN.json', 'w', encoding='UTF-8') as f:
    json.dump(trainData, f)
with open('./data/true-false-qa-TEST.json', 'w', encoding='UTF-8') as f:
    json.dump(testData, f)

print(f'Total nb. of datas = {nbData}')