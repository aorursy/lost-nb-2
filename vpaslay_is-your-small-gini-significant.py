#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv("../input/train.csv")
counts = train.target.value_counts()
print("Class distribution:")
counts / counts.sum()


# In[2]:


import numpy as np
test = pd.read_csv("../input/train.csv")
length = len(test)
np.random.seed(287)
perfect_sub = np.random.rand(length)
target = (perfect_sub > 0.963552).astype(dtype=int)
print("Perfect submission looks like: ", perfect_sub)
print("Target vector looks like: ", target)
print("Target vector class distibution: ")
counts = pd.Series(target).value_counts()
counts / counts.sum()


# In[3]:


from sklearn.metrics import roc_auc_score
def gini(y_target, y_score):
    return 2 * roc_auc_score(y_target, y_score) - 1
print(f"Gini on perfect submission: {gini(target, perfect_sub):0.5f}")


# In[4]:


np.random.seed(8888)
random_sub = np.random.rand(length)
print(f"Gini on random submission: {gini(target, random_sub):0.5f}")


# In[5]:


first30percent = int(length * 0.3)
target_pub = target[:first30percent]
target_prv = target[first30percent:]

def evaluate(submission):
    return gini(target_pub, submission[:first30percent]),        gini(target_prv, submission[first30percent:])


# In[6]:


def spoiler(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    tospoil = np.random.choice(range(length), size=n, replace=False)
    submission = perfect_sub.copy()
    submission[tospoil] = np.random.rand()
    return submission

submissions = []
for spoil_n in range(0, length, 5000):
    score_pub, score_priv = evaluate(spoiler(spoil_n, spoil_n))
    submissions.append((spoil_n, score_pub, score_priv))
submissions = pd.DataFrame(submissions, columns = ["n", "public_score", "private_score"])
submissions.head()


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(11,6))
plt.plot(submissions["n"], submissions["public_score"], label="Public")
plt.plot(submissions["n"], submissions["private_score"], label = "Private")
plt.xlabel("Samples spoiled")
plt.ylabel("Gini Score")
_ = plt.legend()


# In[8]:


import seaborn as sns
pub_prv_diff = submissions["public_score"] - submissions["private_score"]
_ = sns.distplot(pub_prv_diff, hist=True)
_ = plt.xlabel("Public-Private Difference")
_ = plt.ylabel("Density")
pub_prv_diff.describe()


# In[9]:


_ = sns.distplot(abs(pub_prv_diff), bins=20,
             hist_kws=dict(cumulative=-1),
             kde_kws=dict(cumulative=True), kde=False, norm_hist=True)
_ = plt.xlabel("Public-Private Absolute Difference")
_ = plt.ylabel("Frequency")


# In[10]:


np.random.seed(123)
correct = 172560
index = np.random.choice(range(length), size=(length-correct), replace=False)
submission286 = perfect_sub.copy()
spoiled_samples = np.random.rand(length-correct)
submission286[index] = spoiled_samples
public, private = evaluate(submission286)
print(f"Public score: {public:0.4f}\nPrivate score: {private:0.4f}")


# In[11]:


tries, fix = 30, 0
found = False
np.random.seed(10)
while not found:
    fix += 5
    print(f"Fixing {fix} samples")
    for t in range(tries):
        new_submission = submission286.copy()
        improve_index = np.random.choice(index, size=fix, replace=False)
        new_submission[improve_index] = perfect_sub[improve_index]
        public, _ = evaluate(new_submission)
        if public >= 0.287:
            print("0.287 reached!")
            found = True
            break


# In[12]:


# fix_samples contains number of samples to guess
fix_samples = [85, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 
               6000, 7000, 10000, 20000, 30000, 40000, 50000, 60000,
              100000, 200000, 300000, 500000]
# Number of tries for each group of samples
tries = 300
scores, types = [], []
np.random.seed(888)
for fix in fix_samples:
    goal_counter = 0
    # Let's guess and repeat!
    for i in range(tries):
        new_submission = submission286.copy()
        guess = np.random.choice(range(length), size=fix, replace=False)
        new_submission[guess] = np.random.rand()
        public, _ = evaluate(new_submission)
        if public >= 0.287:
            goal_counter += 1
        scores.append(public)
        types.append(fix)
    print(f"Frequency(score>=0.287 | Guessed={fix} samples, Tries={tries} times) = " + 
          f"{goal_counter/tries:0.3f}")
try_history = pd.DataFrame({"type": types, "score": scores})
print("Done!")


# In[13]:


sns.set(font_scale=0.9)
plt.figure(figsize=(15, 10))
ax = sns.stripplot(x="type", y="score", data=try_history, jitter=True, size=9, color="blue", alpha=.5)
ax.set_ylim(0.280, 0.288)
plt.axhline(0.2862, label="baseline", color="blue", linestyle="dashed")
plt.axhline(0.287, label="goal", color="orange", linestyle="dashed")
# Draw means for each group
mean = try_history.groupby("type")["score"].agg(["mean"])["mean"]
plt.plot(range(len(fix_samples)), mean, color="red")
plt.legend()
plt.grid()

