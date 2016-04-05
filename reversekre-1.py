from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph,Sequential
from keras.layers.core import Dense, Dropout,Merge,TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
import cPickle
from sklearn.cross_validation import train_test_split
#from keras.layers.wrappers import Bidirectional

max_features = 20000
maxlen = 150  # cut texts after this number of words (among top max_features most common words)
batch_size = 128
voc_dim=100


print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      #test_split=0.2)

(X, Y) =cPickle.load(open('imdb.pkl', 'rb'))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


index_dict=cPickle.load(open('all.dict.pkl','rb'))
word_vectors=cPickle.load(open('all.vecdict.pkl','rb'))

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


def toa (y):
  #print yl
  for i in range(len(y)):
		y1=np.zeros(3)
		y1[y[i]]=1
		y[i]=y1
  return y

y_train = toa(y_train)
y_test = toa(y_test)

##########################################

(X1, Y1) =cPickle.load(open('emoji.pkl', 'rb'))


eX_train, eX_test, ey_train, ey_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)




print(len(eX_train), 'etrain sequences')
print(len(eX_test), 'etest sequences')

print("Pad sequences (samples x time)")
eX_train = sequence.pad_sequences(eX_train, maxlen=maxlen)
eX_test = sequence.pad_sequences(eX_test, maxlen=maxlen)
print('eX_train shape:', eX_train.shape)
print('eX_test shape:', eX_test.shape)


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
    y1=np.zeros(70)
    y1[yl.index(y[i])]=1
    y[i]=y1
  return y

yl=toyl(ey_train)
ey_train = toaa(ey_train,yl)
ey_test = toaa(ey_test,yl)


def findmax(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=np.zeros(3)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y

def efindmax(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=np.zeros(70)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y


def result(ytest,ypred):
  hit=0
  for i in range(len(ytest)):
    if (ytest[i]==ypred[i]).all() == True:
      hit+=1
  return hit/len(ytest)


print('Build model...')
'''
left=Sequential()
left.add(LSTM(50, input_shape=(maxlen, voc_dim)))

right=Sequential()
right.add(LSTM(50,go_backwards=True, input_shape=(maxlen, voc_dim)))
'''
model = Graph()
model.add_input(name='input1', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(input_dim=n_symbols+1,output_dim=100,mask_zero=True, weights=[embedding_weights]),name='embedding',input='input1')
model.add_node(LSTM(50), name='forward', input='embedding')
model.add_node(LSTM(50, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout1', inputs=['forward', 'backward'])
model.add_output(name='output1', input='dropout1')


#modelf.add(LSTM(50))


model2 = Sequential()
#model2.add_input(name='input1', input_shape=(maxlen,), dtype=int)
model2.add(model)
model2.add(Dense(3, activation='sigmoid'))
#(maxlen,model2.add_output(name='output1', input='sigmoid')
model2.compile('adam', 'binary_crossentropy')


model3 = Sequential()
#model2.add_input(name='input1', input_shape=(maxlen,), dtype=int)
model3.add(model)
model3.add(Dense(70, activation='sigmoid'))
#(maxlen,model2.add_output(name='output1', input='sigmoid')
model3.compile('adam', 'binary_crossentropy')

print('Train...')




for i in np.arange(0,1,0.01):
  l1=len(X_train)
  l2=len(eX_train)
  print(i*100)
  #model2.fit( X_train[np.int(l1*i):np.int(l1*(i+0.1))], y_train[np.int(l1*i):np.int(l1*(i+0.1))],
   #       batch_size=128,
    #      nb_epoch=1,show_accuracy=True)
  model3.fit(eX_train[np.int(l2*i):np.int(l2*(i+0.1))], ey_train[np.int(l2*i):np.int(l2*(i+0.1))],
          batch_size=128,
          nb_epoch=1,show_accuracy=True)

acc = result(y_test,findmax(model3.predict(X_test,batch_size=batch_size)))


print('Test accuracy:', acc)
'''
print('Train...')
model2.fit( X_train, y_train,
          batch_size=batch_size,
          nb_epoch=4,show_accuracy=True)

acc = result(y_test,findmax(model2.predict(X_test,batch_size=batch_size)))


print('Test accuracy:', acc)


print('Train 2...')
model3.fit(eX_train, ey_train,
          batch_size=batch_size,
          nb_epoch=4,show_accuracy=True)

acc = result(y_test,findmax(model2.predict(X_test,batch_size=batch_size)))


print('Test accuracy:', acc)

aacc = result(ey_test,efindmax(model3.predict(eX_test,batch_size=batch_size)))

#y_pred=efindmax(np.array(model3.predict(eX_test,batch_size=batch_size)))

print('Test accuracy:', aacc)
'''