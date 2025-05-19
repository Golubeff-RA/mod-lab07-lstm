import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop

import random
import time
import re

def prepare_text(text):
     text = re.sub(r'[^\w\s]|[\n\r]+', '', text)
     return text.lower()

def pretty_print(text):
    count = 0
    for word in text.split():
        print(word, " ", end="")
        if count % 7 == 0:
            print(" ")
        count += 1

with open('input.txt', 'r') as file:
    text = file.read()

text = prepare_text(text)
print("Число слов в тексте: ", len(text.split()))
pretty_print(text)

# Получаем алфавит
vocabulary = sorted(list(set(text)))
 
# Создаем словари, содержащие индекс символа и связываем их с символами
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

# Разбиваем текст на цепочки длины max_length
# Каждый временной шаг будет загружать очередную цепочку в сеть
max_length = 5
steps = 1
sentences = []
next_chars = []

# Создаем список цепочек и список символов, которые следуют за цепочками
for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])

# Создаем тренировочный набор
# Создаем битовые вектора для входных значений
# (Номер_цепочки-Номер_символа_в цепочке-Код_символа)
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
# Выходные данные
# (Номер_цепочки-Код_символа)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

class LossPlotter(tf.keras.callbacks.Callback):
    def __init__(self, plot_frequency=1):
        super(LossPlotter, self).__init__()
        self.plot_frequency = plot_frequency
        self.losses = []
        self.epochs = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.losses = []
        self.epochs = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.losses.append(logs.get('loss'))
        epoch_time = time.time() - self.start_time 

        if (epoch + 1) % self.plot_frequency == 0:
            self.plot_loss()

    def plot_loss(self):
        plt.clf() 
        plt.plot(self.epochs, self.losses, 'b', label='Training loss')
        plt.title(f'Training Loss (Epoch {self.epochs[-1]})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.pause(0.1)
        plt.show(block=False)


model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)


plotter = LossPlotter()
model.fit(X, y, batch_size = 128, verbose=1, epochs = 50)


def sample_index(preds, temperature = 1000.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
    # Случайное начало
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
 
            generated += next_char
            sentence = sentence[1:] + next_char
    return generated

generated = generate_text(1500, 0.2)
print(len(generated.split()))
pretty_print(generated)
