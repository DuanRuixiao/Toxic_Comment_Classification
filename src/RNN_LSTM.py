import os
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

train = pd.read_csv('../inputs/train.csv')
test = pd.read_csv('../inputs/test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# check tokenizer we used
# for occurence of words
# tokenizer.word_counts
# for index of words
# tokenizer.word_index

# find the proper maxlen value for all sentence
# totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
# plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
# plt.show()

maxlen = 200
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

# Model
inp = Input(shape=(maxlen,))  # maxlen=200 as defined earlier
embed_size = 128

model = Sequential()
model.add(Embedding(max_features, embed_size, input_length=maxlen))
model.add(LSTM(60, return_sequences=True, name='lstm_layer'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(6, activation="sigmoid"))

# model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.summary()

# prediction
pred = model.predict(X_test)
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
y_test = pred.astype(int)
res = pd.DataFrame()
res['id'] = test['id']
res["toxic"] = y_test[:, 0]
res["severe_toxic"] = y_test[:, 1]
res["obscene"] = y_test[:, 2]
res["threat"] = y_test[:, 3]
res["insult"] = y_test[:, 4]
res["identity_hat"] = y_test[:, 5]

os.chdir("../outputs")
res.to_csv("submission_lstm.csv", index=False)
