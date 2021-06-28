
import numpy as np
import json
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout
from bltk.langtools import Tokenizer
from bltk.langtools import remove_stopwords
import random
from keras.optimizers import SGD


############################ Text Processing ##############################
words=[]
classes=[]
documents=[]
data_file=open('intents.json', encoding="utf-8")
tokenizer=Tokenizer()
s=[]
intents=json.load(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = tokenizer.word_tokenizer(pattern)
        w = remove_stopwords(w)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


classes=sorted(list(set(classes)))
words = sorted(list(set(words)))
print(len(words))

training=[]
output_empty=[0]*len(classes)
for doc in documents:
    bag=[]
    pattern_words=doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_vector = list(output_empty)
    output_vector[classes.index(doc[1])] = 1

    training.append([bag, output_vector])
len1=len(classes)
print(classes)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])


#################### Training Model ######################

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add((Dropout(0.4)))
model.add(Dense(32,activation='relu'))
model.add((Dropout(0.4)))
model.add(Dense(len1, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
trained_model=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h2',trained_model)

#model=load_model('chatbot.h1')

############## Predicting ################

while(True):
    msg=input("Write something.......:\n")
    msg_words = tokenizer.word_tokenizer(msg)
    msg_words = remove_stopwords(msg_words)
    bag2=[]
    for w in words:
        if w in msg_words:
            bag2.append(1)
        else:
            bag2.append(0)

    bag2=np.array(bag2)
    y_pred = model.predict_classes(np.array([bag2]))[0]
    output_class=classes[y_pred]
    output_msg=[]
    for i in intents['intents']:
        if i['tag'] == output_class:
            output_msg=random.choice(i['responses'])
            break
    print(output_msg)

