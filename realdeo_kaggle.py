import pandas as pd 

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

print(train.shape)
print(train.head(n=5))
print(test.shape)
print(test.head(n=5))


target=train['Response']
bmiOver=[int(0) for i in range(8)]
bmiNotOver=[int(0) for i in range(8)]
for i in range(len(train)):
    number=train['BMI'][i]
    if number>0.5:
        bmiOver[target[i]-1]+=1
    else:
        bmiNotOver[target[i]-1]+=1
print(bmiOver)
print(bmiNotOver)
print(number)

from sklearn.linear_model import LogisticRegression

LR=LogisticRegression

LR.fit([train['BMI'],train[', target, sample_weight=None)
