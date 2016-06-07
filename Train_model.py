from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1337) # for reproducibility
import os,h5py
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,RepeatVector
import cPickle
from sklearn.cross_validation import train_test_split
from collections import Counter as C


max_features = 20000
maxlen = 140  
batch_size = 1
voc_dim=100


print('Loading data...')

#stsgold-all.pkl
#sem-all.pkl
#semeval-all-train.pkl
(X, Y) =cPickle.load(open('./pkl/semeval-all.pkl', 'rb'))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#load directionary and word embedding weights

index_dict=cPickle.load(open('emoji-all.dict.pkl','rb'))
word_vectors=cPickle.load(open('emoji-all.vecdict.pkl','rb'))

n_symbols=len(index_dict)+1
embedding_weights=np.zeros((n_symbols+1,voc_dim))
for w,i in index_dict.items():
	embedding_weights[i,:]=word_vectors[w]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def to3 (y):
  #print yl
  for i in range(len(y)):
    y1=zerolistmaker(2)
    y1[y[i]]=1
    y[i]=y1
  return y



y_train = to3(y_train)
y_test = to3(y_test)


(X2, Y2) =cPickle.load(open('emoji-all.pkl', 'rb'))


eeX_train, eeX_test, eey_train, eey_test = train_test_split(X2[:600000], Y2[:600000], test_size=0.2, random_state=42)




print(len(eeX_train), 'etrain sequences')
print(len(eeX_test), 'etest sequences')

print("Pad sequences (samples x time)")
eeX_train = sequence.pad_sequences(eeX_train, maxlen=maxlen)
eeX_test = sequence.pad_sequences(eeX_test, maxlen=maxlen)
print('eX_train shape:', eeX_train.shape)
print('eX_test shape:', eeX_test.shape)



def toyl(y):
  yl=[]
  for y0 in y:
    if y0 not in yl:
      yl.append(y0)
    else:
      continue
  return yl
def toaa (y,yl):
  #print yl
  for i in range(len(y)):
    y1=zerolistmaker(70)
    y1[yl.index(y[i])]=1
    y[i]=y1
  return y

yl=toyl(eey_train)
print (yl)
eey_train = toaa(eey_train,yl)
eey_test = toaa(eey_test,yl)

'''
List of emojis
[u'\U0001f606', u'\U0001f62d', u'\U0001f602', u'\U0001f615', u'\U0001f60d', u'\U0001f629', u'\U0001f60a', u'\U0001f644', u'\U0001f642', u'\U0001f601', u'\U0001f914', u'\U0001f609', u'\U0001f611', u'\U0001f643', u'\U0001f614', u'\U0001f621', u'\U0001f612', u'\U0001f60f', u'\U0001f60c', u'\U0001f622', u'\U0001f637', u'\U0001f60e', u'\U0001f616', u'\U0001f600', u'\U0001f61e', u'\U0001f60b', u'\U0001f608', u'\U0001f610', u'\U0001f618', u'\U0001f641', u'\U0001f62c', u'\U0001f62f', u'\U0001f61c', u'\u263a\ufe0f', u'\U0001f47f', u'\U0001f605', u'\U0001f624', u'\U0001f633', u'\u2639\ufe0f', u'\U0001f917', u'\U0001f613', u'\U0001f604', u'\U0001f620', u'\U0001f636', u'\U0001f628', u'\U0001f61b', u'\U0001f62b', u'\U0001f61f', u'\U0001f627', u'\U0001f61d', u'\U0001f631', u'\U0001f623', u'\U0001f634', u'\U0001f62a', u'\U0001f603', u'\U0001f635', u'\U0001f911', u'\U0001f915', u'\U0001f913', u'\U0001f607', u'\U0001f62e', u'\U0001f61a', u'\U0001f910', u'\U0001f632', u'\U0001f912', u'\U0001f625', u'\U0001f626', u'\U0001f630', u'\U0001f619', u'\U0001f617']

'''



def findmax(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=zerolistmaker(3)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y

def findmax2(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=zerolistmaker(3)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y

def efindmax(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=zerolistmaker(70)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y


def result(ytest,ypred):
  hit=0
  cp=[]
  wp={}
  m=np.zeros( (len(ytest[0]),len(ytest[0])) )
  for i in range(len(ytest)):
    if (ytest[i]==ypred[i]).all() == True:
      hit+=1
      cp.append(ytest[i].index(1))
    m[ytest[i].index(1)][ypred[i].nonzero()[0]]+=1
    #else:
     # if ytest[i].index(1) in wp.keys():
      #  wp[ytest[i].index(1)]=wp[ytest[i].index(1)]+ypred[i].nonzero()[0]
      #else:
       # wp[ytest[i].index(1)]=ypred[i].nonzero()[0]
  #for w in wp.keys():
    #wp[w]=C(wp[w])
  return hit/len(ytest),C(cp),m


print('Build model...')

In2 = Input(shape=(maxlen,), dtype='int32')
m2 = Embedding(input_dim=n_symbols+1,output_dim=100,mask_zero=True, weights=[embedding_weights])(In2)
f2 = LSTM(100)(m2)
b2 = LSTM(100, go_backwards=True)(m2)
me2 = merge([f2, b2], mode='concat', concat_axis=-1)
dp2 = Dropout(0.5)(me2)

d2 = Dense(3, activation='softmax')(dp2)
d4 = Dense(70, activation='softmax')(dp2)
model4 = Model(input=In2,output=d4)
model4.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model2 = Model(input=In2,output=d2)
model2.compile('adam', 'binary_crossentropy', metrics=['accuracy'])





print('Train...')



for i in np.arange(0,1,0.1):
  l1=len(X_train)
  l3=len(eeX_train)
  model4.fit(eeX_train[np.int(l3*i):np.int(l3*(i+0.0150))], eey_train[np.int(l3*i):np.int(l3*(i+0.0150))],batch_size=batch_size,nb_epoch=1)
  model2.fit( X_train[np.int(l1*i):np.int(l1*(i+0.1))], y_train[np.int(l1*i):np.int(l1*(i+0.1))],batch_size=batch_size,nb_epoch=1)
  

#model2.load_weights('model2.h5')
#model3.load_weights('model3.h5')
#model4.load_weights('model4.h5')

model2.save_weights('model2-0.0150-n28.h5')
#model3.save_weights('model3.h5')
model4.save_weights('model4-0.0150-n28.h5')

acc,cp1,wp1 = result(y_test,findmax(model2.predict(X_test,batch_size=batch_size)))
print('Test accuracy:', acc)
print('corrent',cp1)
print('Matrix\n',wp1)
#aacc,cp2,wp2 = result(ey_test,findmax2(model3.predict(eX_test,batch_size=batch_size)))
#print('Test accuracy:', aacc)

eey_test=eey_test[:1600]
eeX_test=eeX_test[:1600]

aaccaa,cp3,wp3 = result(eey_test,efindmax(model4.predict(eeX_test,batch_size=batch_size)))
print('Test accuracy:', aaccaa)
print('corrent',cp3)
print('Matrix\n',wp3)

os.system('say ee wan cheng')
