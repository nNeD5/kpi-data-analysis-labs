# pyright: basic
# %% [markdown]
r"""
# Lab Work #3
### Nedozhdii Oleksii FF-31mn
"""
# %% [markdown]
r"""
# Imports
"""
# %%

import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

pio.templates.default = "plotly_dark"
pio.renderers.default = "jupyterlab"

# %% [markdown]
r"""
# 1. Повнозв'язані нейронні мережі.
Вирішіть завдання класифікації даних, з якими ви працювали в лабораторній №1 за допомогою повнозв’язаної нейромережі прямого поширення (fully connected feed-forward network). Результати порівняйте з одержаними раніше.
"""
# %% [markdown]
r"""
## Load and prepare dataset 'OnlineNewsPopularity'
"""
# %%

online_news_df = pd.read_csv("resources/OnlineNewsPopularity/OnlineNewsPopularity.csv",
                 skipinitialspace=True)
online_news_df = online_news_df.drop(columns=["url"])

# choose target and features
targets = [
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]
online_news_df = online_news_df.dropna()
online_news_df = online_news_df[online_news_df[targets].any(axis=1)]

labels = dict(zip(targets, ["Lifestyle", "Entertainment", "Business", "Social Media", "Tech", "World"]))
online_news_df["data_channel"] = online_news_df.apply(lambda row: [col for col in online_news_df.columns if row[col] and col.startswith("data_channel_")][0], axis=1)
for key in labels.keys():
    online_news_df.loc[online_news_df["data_channel"] == key, "data_channel"] = labels[key]
online_news_df = online_news_df.drop(columns=[col for col in online_news_df.columns if col.startswith("data_channel_is")])
del targets

target = "data_channel"
features = online_news_df.columns.drop(labels=target).to_list()

print(f"Target: {target}")
print(f"Features: {features}")

# %%

le = LabelEncoder()
online_news_df[target] = le.fit_transform(online_news_df[target])

# %%

scaler = StandardScaler()
features_std = pd.DataFrame(scaler.fit_transform(online_news_df[features]), columns=features)
X_online_news, y_online_news = features_std, online_news_df[target]

# %%

online_news_X_train, online_news_X_test, online_news_y_train, online_news_y_test = train_test_split(
        X_online_news, y_online_news,
        test_size=0.2, random_state=42
)

#%% [markdown]
r"""
## Random forest
"""
# %%

random_forest = RandomForestClassifier(class_weight="balanced", criterion="gini", n_estimators=250)
random_forest.fit(online_news_X_train, online_news_y_train)
# %%

y_predict_random_forest = random_forest.predict(online_news_X_test)

random_forest_conf = confusion_matrix(online_news_y_test, y_predict_random_forest)
random_forest_report = classification_report(online_news_y_test, y_predict_random_forest)
random_forest_score = accuracy_score(online_news_y_test, y_predict_random_forest)

#%% [markdown]
r"""
## Fully connected feed-forward network
"""
# %%

oline_news_num_classes = len(online_news_df[target].unique())

fc_network = keras.Sequential()
fc_network.add(layers.Input((online_news_X_train.shape[1], )))
fc_network.add(layers.Dense(500, activation="relu"))
fc_network.add(layers.Dense(400, activation="relu"))
fc_network.add(layers.Dense(300, activation="relu"))
fc_network.add(layers.Dense(250, activation="relu"))
fc_network.add(layers.Dense(oline_news_num_classes, activation="softmax"))

fc_network.summary()
fc_network.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%

fc_history = fc_network.fit(online_news_X_train, online_news_y_train,
                    epochs=13, batch_size=online_news_X_train.shape[0]//7,
                    validation_split=0.2)
# %%

fc_loss_values = fc_history.history["loss"]
fc_val_loss_values = fc_history.history["val_loss"]
fc_epochs = list(range(1, len(fc_loss_values) + 1))

fc_score = fc_network.evaluate(online_news_X_test, online_news_y_test, verbose=0)
print("Test score:", fc_score[0])
print("Test accuracy:", fc_score[1])

# %%

fig = go.Figure()
fig.add_trace(go.Scatter(x=fc_epochs, y=fc_loss_values, name="Training loss"))
fig.add_trace(go.Scatter(x=fc_epochs, y=fc_val_loss_values, mode="markers", name="Validation loss"))
fig.update_layout(autosize=True, title="Training and validation loss")
fig.update_xaxes(title="Epochs")
fig.update_yaxes(title="Loss")
fig.show()

# %%

y_predict_fc = fc_network.predict(online_news_X_test)
y_predict_fc = np.argmax(y_predict_fc, axis=1)

fc_network_score = accuracy_score(online_news_y_test, y_predict_fc)
fc_network_conf = confusion_matrix(online_news_y_test, y_predict_fc)
fc_network_report = classification_report(online_news_y_test, y_predict_fc)

# %% [markdown]
r"""
## Random forest vs Fully connected feed-forward network

winner: Random Forest
"""
# %%

print("====== Random forest ======")
print(random_forest_score)
print(random_forest_conf)
print(random_forest_report)

print()

print("====== Fully connected feed-forward network ======")
print(fc_network_score)
print(fc_network_conf)
print(fc_network_report)

print("====== Score diff: RF - FCFFN ======")
print(random_forest_score - fc_network_score)
