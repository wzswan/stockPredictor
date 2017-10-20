from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
# this optimizers should be replced
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('')
data = open(path).read()
print('data length:', len(data))

chars = sorted(list(set(data)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the data in semi-redundant sequences of maxlen charaters
maxlen = 10
step = 3
sentences = []
next_char = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_char.append(data[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences),len(chars)), dtype=np.bool)
for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
modle.add(Activation('softmax'))

optimizer = RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, tempreature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / tempreature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated data after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)
    start_index = random.randint(0, len(data) - maxlen -1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('------diversity:', diversity)

        generated = ''
        sentence = data[start_index: start_index + maxlen]
        generated += sentence
        print('------Generating with seed:"'+ sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1self.

            preds = model.preict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indiecs_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
