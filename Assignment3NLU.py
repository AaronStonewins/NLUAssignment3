
# coding: utf-8

# In[1]:


import sklearn_crfsuite
from sklearn_crfsuite import metrics


# In[2]:


s='./Desktop/ner.txt'


# In[3]:


trainwords=[];testwords=[];totalwords=[]
with open(s,encoding='latin1') as f:
    sentence = []
    for line in f:
        if line != "\n":
            x=line.strip("\n").split()
            sentence.append(tuple(x))
        else:
            totalwords.append(sentence)
            sentence = []


# In[4]:


trainwords=totalwords[0:int(0.7*len(totalwords))]
testwords=totalwords[int(0.7*len(totalwords)):len(totalwords)]


# In[5]:


def wtofeatures(sentence, i):
    word = sentence[i][0]

    features = {
        'bias': 2.0,
        'lower': word.lower(),
        'begtothirdlastword': word[-3:],
        'begtosecondlastword': word[-2:],
        'upperornot': word.isupper(),
        'titleornot': word.istitle(),
        'numberornot': word.isdigit(),
    }
    if i > 0:
        wordsub= sentence[i-1][0]
        features.update({
            'lower_-1': wordsub.lower(),
            'titleornot_-1': wordsub.istitle(),
            'upperornot_-1': wordsub.isupper(),
        })
    else:
        features['<s>'] = True

    if i < len(sentence)-1:
        wordsub = sentence[i+1][0]
        features.update({
            'lower_+1': wordsub.lower(),
            'titleornot_+1': wordsub.istitle(),
            'upperornot_+1': wordsub.isupper(),
        })
    else:
        features['<\s>'] = True

    return features


def stofeatures(sentence):
    return [wtofeatures(sentence, i) for i in range(len(sentence))]

def stolabels(sentence):
    return [label for token,label in sentence]

def stotokens(sentence):
    return [token for token,label in sentence]


# In[6]:


Xtrain = [stofeatures(s) for s in trainwords]
ytrain = [stolabels(s) for s in trainwords]

Xtest = [stofeatures(s) for s in testwords]
ytest = [stolabels(s) for s in testwords]


# In[7]:


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.15,
    c2=0.15,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(Xtrain, ytrain)


# In[8]:


labels = list(crf.classes_)
predicted = crf.predict(Xtest)
metrics.flat_f1_score(ytest, predicted,
                      average='weighted', labels=labels)


# In[9]:


labelsorted = sorted(
    labels,
    key=lambda n: (n[1:], n[0])
)
print(metrics.flat_classification_report(
    ytest, predicted, labels=labelsorted, digits=3
))

