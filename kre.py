'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''
from __future__ import division
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
import cPickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

max_features = 20000
maxlen = 150  # cut texts after this number of words (among top max_features most common words)
batch_size = 256
voc_dim=100


print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      #test_split=0.2)

(X, Y) =cPickle.load(open('emoji.pkl', 'rb'))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


index_dict=cPickle.load(open('emoji.dict.pkl','rb'))
word_vectors=cPickle.load(open('emoji.vecdict.pkl','rb'))

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

def toyl(y):
  yl=[]
  for y0 in y:
    if y0 not in yl:
      yl.append(y0)
    else:
      continue
  return yl
def toa (y,yl):
  #print yl
  for i in range(len(y)):
		y1=np.zeros(70)
		y1[yl.index(y[i])]=1
		y[i]=y1
  return y

def findmax(y):
  for yi in range(len(y)):
    #m=max(y[yi])
    y1=np.zeros(70)
    y1[np.argmax(y[yi])]=1
    y[yi]=y1
  return y


def result(y_test,y_pred):
  hit=0
  for i in range(len(y_test)):
    if (y_test[i]==y_pred[i]).all() == True:
      hit+=1
  return hit/len(y_test)


yl=toyl(y_train)
#mlb=MultiLabelBinarizer()
y_train = toa(y_train,yl)
y_test = toa(y_test,yl)
#y_train=mlb.fit_transform(y_train)
#y_test=mlb.fit_transform(y_test)

print('Build model...')
model = Graph()
model.add_input(name='input', input_shape=(maxlen,), dtype=int)
model.add_node(Embedding(input_dim=n_symbols+1,output_dim=100,mask_zero=True, weights=[embedding_weights]),
               name='embedding',input='input')
model.add_node(LSTM(50), name='forward', input='embedding')
model.add_node(LSTM(50, go_backwards=True), name='backward', input='embedding')
model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
model.add_node(Dense(70, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')

# try using different optimizers and different optimizer configs
model.compile('adam', {'output': 'binary_crossentropy'})

print('Train...')
model.fit({'input': X_train, 'output': y_train},
          batch_size=batch_size,
          nb_epoch=40)

'''
acc = accuracy(y_test,
               model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])
'''
acc = result(y_test,
               findmax(np.array(model.predict({'input': X_test},
                                               batch_size=batch_size)['output'])))
y_pred=findmax(np.array(model.predict({'input': X_test},batch_size=batch_size)['output']))

print('Test accuracy:', acc)





