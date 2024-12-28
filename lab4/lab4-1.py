# pyright: basic
# %% [markdown]
r"""
Недождій Олексій ФФ-31мн

Lab 4. Advanced Nets
1. Завдання щодо генерації текстів або машинного перекладу (на вибір) на базі рекурентних мереж або трансформерів (на вибір).
Вирішіть завдання щодо генерації текстів або машинного перекладу. Особливо вітаються україномовні моделі.

generate text character-by-character
dataset: https://www.kaggle.com/datasets/mykras/ukrainian-texts
"""

# %%
import numpy as np
import os
import keras
from keras import layers
import random

# %%
root = "resources"
text = ""
# TODO: clean space
for file in os.listdir(root):
    with open(os.path.join(root, file)) as f:
        text += f.read().lower().replace("\n", " ")
print(f"Text lenght: {len(text)}")
# text = text[:10000]

# %%
chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# %%
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

# %%
x = np.zeros((len(sentences), maxlen, len(chars)), dtype="bool")
y = np.zeros((len(sentences), len(chars)), dtype="bool")
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# %%
model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))), # ?????
        layers.LSTM(128), # ?????
        layers.Dense(len(chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)  # ?????
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# %%
def sample(preds, temperature=1.0):
    # quasi activation layer
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(n=1, pvals=preds, size=1)
    return np.argmax(probas)


# %%
model.fit(x, y, batch_size=128, epochs=10)

# %%
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print("\033[32mDiversity:\033[0m ", diversity)

    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ""
    sentence = text[start_index : start_index + maxlen]
    print('\033[32mGenerating with seed:\033[0m "' + sentence + '"')

    for i in range(maxlen):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
    print("\033[32mGenerated:\033[0m ", generated)
    print("-")
