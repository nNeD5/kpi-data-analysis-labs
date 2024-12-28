# pyright: basic
# %% [markdown]
r"""
# Lab Work #3
### Nedozhdii Oleksii FF-31mn
"""

# %% [markdown]
r"""°°°
# Imports
°°°"""
# %%

import os
import random
import re
import string

import keras
import numpy as np
import splitfolders
import tensorflow as tf
import torch
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# %% [markdown]
r"""
# 3. Рекурентні нейронні мережі
Вирішіть задачу класифікації текстів (з якими ви працювали в лабораторній № 2) за допомогою рекурентної нейромережі двома способами:
а) навчить мережу і embedding шар з нуля (from scratch)
б) використовуючи pretrained word embeddings
"""

# %% [markdown]
r"""°°°
## Load and prepare text data
°°°"""
# %%

multimodal_path = "resources/multimodal"
multimodal_path_splited = "resources/multimodal/splited"

# %%

# splitfolders.ratio(multimodal_path, output=multimodal_path_splited,
#                    seed=42, ratio=(.8, .1, .1))

# %%

raw_txt_train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(multimodal_path_splited, "train"), batch_size=16, seed=42
)
raw_txt_val_ds = keras.utils.text_dataset_from_directory(
    os.path.join(multimodal_path_splited, "val"), batch_size=16, seed=42, shuffle=False
)
raw_txt_test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(multimodal_path_splited, "test"), batch_size=16, seed=42, shuffle=False
)

# %%

txt_class_names = raw_txt_test_ds.class_names
txt_class_number = len(txt_class_names)
print(txt_class_names)
print(txt_class_number)

# %%


def datatset_standardization(text, label):
    # print(raw_txt)
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(
        text, r"@\S+", " "
    )  # remove mentions
    text = tf.strings.regex_replace(
        text, r"https*\S+", " "
    )  # remove url
    text = tf.strings.regex_replace(
        text, r"#\S+", " "
    )  # remove hashtags
    text = tf.strings.regex_replace(
        text, r"\d", " "
    )  # remove all numbers
    # remove punctuations
    text = tf.strings.regex_replace(
        text, r"[%s]" % re.escape(string.punctuation), " "
    )
    text = tf.strings.regex_replace(
        text, r"\s{2,}", " "
    )  # remove extra spaces
    text = tf.strings.regex_replace(
        text, r"[^\x00-\x7F]+", ""
    )  # remove not ascii
    text = tf.strings.strip(text)
    return (text, label)

formated_txt_train_ds = raw_txt_train_ds.map(datatset_standardization)
formated_txt_test_ds = raw_txt_test_ds.map(datatset_standardization)
formated_txt_val_ds = raw_txt_val_ds.map(datatset_standardization)

formated_txt_train_ds = formated_txt_train_ds.unbatch().filter(lambda x, y: tf.strings.length(x) > 0)
formated_txt_test_ds = formated_txt_test_ds.unbatch().filter(lambda x, y: tf.strings.length(x) > 0)
formated_txt_val_ds = formated_txt_val_ds.unbatch().filter(lambda x, y: tf.strings.length(x) > 0)

formated_txt_train_ds = formated_txt_train_ds.batch(16)
formated_txt_test_ds = formated_txt_test_ds.batch(16)
formated_txt_val_ds = formated_txt_val_ds.batch(16)
# %%

for item in formated_txt_train_ds.unbatch():
    if item[0].numpy().decode("utf-8") == "":
        print("Spot emtpy line")

# %%

i = raw_txt_train_ds.unbatch().as_numpy_iterator()
print(next(i), end="\n\n")
print(next(i), end="\n\n")
print(next(i), end="\n\n")
del i

# %%

i = formated_txt_train_ds.as_numpy_iterator()
print(next(i), end="\n\n")
print(next(i), end="\n\n")
print(next(i), end="\n\n")
del i

# %%

max_features = 10000
sequence_length = 500

vectorize_layer = layers.TextVectorization(
    # standardize=datatset_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
train_text = formated_txt_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# %%

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# %%

train_vec_txt_ds = formated_txt_train_ds.map(vectorize_text)
val_vec_txt_ds = formated_txt_val_ds.map(vectorize_text)
test_vec_txt_ds = formated_txt_test_ds.map(vectorize_text)

print(next(train_vec_txt_ds.as_numpy_iterator()))

# %%

AUTOTUNE = tf.data.AUTOTUNE

train_txt_ds = train_vec_txt_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_txt_ds = val_vec_txt_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_txt_ds = test_vec_txt_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
r"""°°°
## Model from scratch
°°°"""
# %%

emb_network = keras.Sequential()
emb_network.add(layers.Embedding(max_features, 16))
emb_network.add(layers.Dropout(0.2))
emb_network.add(layers.GlobalMaxPooling1D())
emb_network.add(layers.Dropout(0.2))
emb_network.add(layers.Dense(6, activation="softmax"))

emb_network.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
emb_network.summary()

# %%

history_emb_network = emb_network.fit(
    train_txt_ds, validation_data=val_txt_ds, epochs=25
)

# %%

emb_loss, emb_accuracy = emb_network.evaluate(test_txt_ds)

print("Test loss:", emb_loss)
print("Test accuracy:", emb_accuracy)

# %%

txt_true_labels = []

for _, label in test_txt_ds.unbatch():
    txt_true_labels.append(label.numpy())

# %%

y_predict_emb = emb_network.predict(test_txt_ds)
y_predict_emb = np.argmax(y_predict_emb, axis=1)
emb_network_score = accuracy_score(txt_true_labels, y_predict_emb)
emb_network_conf = confusion_matrix(txt_true_labels, y_predict_emb)
emb_network_report = classification_report(txt_true_labels, y_predict_emb)

print(f"Score: {emb_network_score}")

# %% [markdown]
r"""°°°
## Bert model
°°°"""
# %%

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=txt_class_number
)

# %%
formated_txt_bert_ds = formated_txt_train_ds.unbatch()
text = formated_txt_bert_ds.map(lambda x, y: x)
text = list(np.fromiter(text.as_numpy_iterator(), dtype=((str, 512))))[:20]
txt_vec_bert_ds = bert_tokenizer(text, truncation=True, padding=True, return_tensors="pt")

# %%

bert_model.eval()
bert_output = bert_model(**txt_vec_bert_ds)

# %%

predictions = tf.nn.softmax(bert_output[0].detach(), axis=-1)

# %%

formated_txt_test_list = list(formated_txt_bert_ds.as_numpy_iterator())[:20]
print(formated_txt_test_list[0])
print([txt_class_names[i[1]] for i in formated_txt_test_list])

print(txt_class_names)
for i in range(20):
    print(f"\033[93mSource:\033[0m {text[i]}")
    true_class_code = formated_txt_test_list[i][1]
    pred_class_code = np.argmax(predictions[i])
    print(f"\033[32mtrue:\033[0m {txt_class_names[true_class_code]} | \033[31mpred:\033[0m {txt_class_names[pred_class_code]}")
    print()
