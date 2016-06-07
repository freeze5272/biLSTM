from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1337) # for reproducibility
import os,h5py,glob
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,RepeatVector
import cPickle as pkl
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from collections import Counter as C
import matplotlib.pyplot as plt


max_features = 20000
maxlen = 140  
batch_size = 1
voc_dim=100
#load directionary and word embedding weights
index_dict=pkl.load(open('emoji-all.dict.pkl','rb'))
word_vectors=pkl.load(open('emoji-all.vecdict.pkl','rb'))

n_symbols=len(index_dict)+1
embedding_weights=np.zeros((n_symbols+1,voc_dim))
for w,i in index_dict.items():
	embedding_weights[i,:]=word_vectors[w]



def load_data(d):
	(X, Y) =pkl.load(open(d, 'rb'))
	

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	el=[u'\U0001f606', u'\U0001f62d', u'\U0001f602', u'\U0001f615', u'\U0001f60d', u'\U0001f629', u'\U0001f60a', u'\U0001f644', u'\U0001f642', u'\U0001f601', u'\U0001f914', u'\U0001f609', u'\U0001f611', u'\U0001f643', u'\U0001f614', u'\U0001f621', u'\U0001f612', u'\U0001f60f', u'\U0001f60c', u'\U0001f622', u'\U0001f637', u'\U0001f60e', u'\U0001f616', u'\U0001f600', u'\U0001f61e', u'\U0001f60b', u'\U0001f608', u'\U0001f610', u'\U0001f618', u'\U0001f641', u'\U0001f62c', u'\U0001f62f', u'\U0001f61c', u'\u263a\ufe0f', u'\U0001f47f', u'\U0001f605', u'\U0001f624', u'\U0001f633', u'\u2639\ufe0f', u'\U0001f917', u'\U0001f613', u'\U0001f604', u'\U0001f620', u'\U0001f636', u'\U0001f628', u'\U0001f61b', u'\U0001f62b', u'\U0001f61f', u'\U0001f627', u'\U0001f61d', u'\U0001f631', u'\U0001f623', u'\U0001f634', u'\U0001f62a', u'\U0001f603', u'\U0001f635', u'\U0001f911', u'\U0001f915', u'\U0001f913', u'\U0001f607', u'\U0001f62e', u'\U0001f61a', u'\U0001f910', u'\U0001f632', u'\U0001f912', u'\U0001f625', u'\U0001f626', u'\U0001f630', u'\U0001f619', u'\U0001f617']


	print(d[6:-4],'dateset')
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

	'''
	if d == './pkl/emoji-all.pkl':
		y_test = y_test[:1600]
		y_test = to70(y_test,el)
		X_test = X_test[:1600]

	else:
		#Y= to2(Y)
		y_test = to2(y_test)
	'''
	y_test = to2(y_test)
	#return X,Y
	return X_test,y_test

#formating for Y 
def zerolistmaker(n):
		listofzeros = [0] * n
		return listofzeros

def to2 (y):
	#print yl
	for i in range(len(y)):
		y1=zerolistmaker(2)
		y1[y[i]]=1
		y[i]=y1
	return y

def to70 (y,el):
  #print yl
  for i in range(len(y)):
    y1=zerolistmaker(70)
    y1[el.index(y[i])]=1
    y[i]=y1
  return y

#formatring for result
def findmax2(y):
	for yi in range(len(y)):
		#m=max(y[yi])
		y1=zerolistmaker(2)
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


def result(ytest,ypred,n=0):
	hit=0
	cp=[]
	wp={}
	f1=[]
	m=np.zeros( (len(ytest[0]),len(ytest[0])) )
	for i in range(len(ytest)):
		if (ytest[i]==ypred[i]).all() == True:
			hit+=1
			cp.append(ytest[i].index(1))
		m[ytest[i].index(1)][ypred[i].nonzero()[0]]+=1
	#M is matrix for calculating F1 score
	for m1 in range(len(m)):
		tp=m[m1][m1]
		fp=sum(m[m1,:])
		fn=sum(m[:,m1])

		p=tp/fp
		r=tp/fn
		f1s=2*p*r/(p+r)
		'''
		#For checking F1 score
		print ('tp:',tp)
		print ('fp',fp)
		print ('fn',fn)
		print ('p',p)
		print ('r',r)
		print ('f1s',f1s)
		'''
		f1.append(f1s)
	if n==1:
		mf1=np.mean(f1[:-1])
	if n==0:
		mf1=np.mean(f1)
		'''
	#print(hit/len(ytest))
	#print(f1[:-1])
		#else:
		 # if ytest[i].index(1) in wp.keys():
			#  wp[ytest[i].index(1)]=wp[ytest[i].index(1)]+ypred[i].nonzero()[0]
			#else:
			 # wp[ytest[i].index(1)]=ypred[i].nonzero()[0]
	#for w in wp.keys():
		#wp[w]=C(wp[w])
		'''
	return mf1#hit/len(ytest)#np.mean(f1)#hit/len(ytest)#,C(cp),m

In2 = Input(shape=(maxlen,), dtype='int32')
m2 = Embedding(input_dim=n_symbols+1,output_dim=100,mask_zero=True, weights=[embedding_weights])(In2)
f2 = LSTM(100)(m2)
b2 = LSTM(100, go_backwards=True)(m2)
me2 = merge([f2, b2], mode='concat', concat_axis=-1)
dp2 = Dropout(0.5)(me2)

d2 = Dense(2, activation='softmax')(dp2)
d4 = Dense(70, activation='softmax')(dp2)
model4 = Model(input=In2,output=d4)
model4.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model2 = Model(input=In2,output=d2)
model2.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
m2=glob.glob('./n28model/model2-*.h5')
m4=glob.glob('./n28model/model4-*.h5')
#m2=glob.glob('./n6kmodel/model2-*.h5')
#m4=glob.glob('./n6kmodel/model4-*.h5')
#m2=glob.glob('./nallmodel/model2-*.h5')
#m4=glob.glob('./nallmodel/model4-*.h5')

yl=[int(float(x[-9:-3])*10*480000) for x in m2]
print (yl,len(yl))

p=open('N28.pkl','wb')
fl=glob.glob('./pkl/*')
xl=[]
for ff in fl:
	X,Y=load_data(ff)
	al=[]
	for i in range(len(m2)):
		model2.load_weights(m2[i])
		model4.load_weights(m4[i])
		if ff != './pkl/emoji-all.pkl':
			acc = result(Y,findmax2(model2.predict(X,batch_size=batch_size)))
		else:
			acc = result(Y,efindmax(model4.predict(X,batch_size=batch_size)))
		#print (acc)
		al.append(acc)
	xl.append(al)
	print (al[0],'first')
	print (al[-1],'last',al[-1]-al[0],'change')
	print (max(al),'max',al.index(max(al)),'position',max(al)-al[0],'change')
	print (al,'\n')
	plt.plot(yl,al)
#print(al)
#plt.plot(al)
pkl.dump((xl,yl),p)
p.close()
plt.legend([x[6:-4] for x in fl], loc='lower left')
plt.xlabel('Number of tweets')
plt.ylabel('F1 score')
plt.xticks(yl)
os.system('say ee wan cheng')
plt.show() 

