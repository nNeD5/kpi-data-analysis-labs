# %% [markdown]
# # Lab Work #1
# ### Nedozhdii Oleksii FF-31mn
#
#  1) Завантажити дані, вивести назви колонок і розмір датасета
#  2) Опрацювати пропуски (по можливості заповнити їх або видалити)
#  3) Візуалізувати дані: побудувати графік (heatmap), що відображає кореляції
#  ознак між собою і з цільовою змінною (розміткою); побудувати гістограми
#  розподілу ознак і boxplot-и ознак відносно цільової змінної (якщо ознак занадто багато
#  обмежитися декількома)
#  4) Нормалізувати дані
#  5) Провести навчання наступних класифікаторів:
#  - kNN
#  - дерево ухвалення рішень
#  - SVM
#  - Random Forest
#  - AdaBoost
#
#  Підібрати оптимальні параметри
#  - для kNN
#  - для SVM за допомогою GridSearch підібрати оптимальні `C` і `gamma`
#  - Серед обраних оптимальних моделей кожного класу вибрати найкращу.
#  - Відобразити
#  sklearn.metrics.classification_report і sklearn.metrics.confusion_matrix

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV


plt.style.use("default")

# %% [markdown]
# ### Preate data

# %%
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv",
                 skipinitialspace=True)

# it's not fair to compare shares number for 8 and 30 days articles
# drop not useful(for now) data
df["shares_per_day"] = df["shares"] / df["timedelta"]
df = df.drop(columns=["url", "timedelta", "shares",
                      "is_weekend", "LDA_00", "LDA_01",
                      "LDA_02", "LDA_03", "LDA_04"])

# choose target and features
targets=[
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]
features = df.columns.drop(labels=targets)

# delete rows without target values
df = df[df[targets].any(axis=1)]

# check and drop null valus
null_indexes = sum(np.nonzero(pd.isnull(df)))
if len(null_indexes) > 0:
    print("Ups, spot null value. Going to drop it.")
    df = df.dropna()

# multiple columns data_channel_is_* -> one column with multiple values
labels = dict(zip(targets, ["Lifestyle", "Entertainment", "Business", "Social Media", "Tech", "World"]))
df["data_channel"] = df.apply(lambda row: [col for col in df.columns if row[col] and col.startswith("data_channel_")][0], axis=1)
for key in labels.keys():
    df.loc[df["data_channel"] == key, "data_channel"] = labels[key]
df = df.drop(columns=[col for col in df.columns if col.startswith("data_channel_is")])
del targets

# %% [markdown]
# ### Present data

# %%
df.info()
df.describe()

# %% [markdown]
# ### Features and target

# %%
target = "data_channel"
features = df.columns.drop(labels=target).to_list()
target_values = df["data_channel"].unique()

# %% [markdown]
# ### Calculate correlation matrix

# %%
le = LabelEncoder()
df_encoded = df.copy()
df_encoded[target] = le.fit_transform(df[target])
corr_matrix = df_encoded.corr()

# %% [markdown]
# ### Heatmap All

# %%
fig, ax = plt.subplots(layout="constrained", figsize=(16, 8))
fig.suptitle("Correlations of attributes", fontsize=26)
im = ax.imshow(corr_matrix, aspect="auto")
ax.set_xticks(np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, fontsize=12)
ax.set_yticks(np.arange(len(corr_matrix.index)), labels=corr_matrix.index, fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.colorbar(im);

# %% [markdown]
# ### Heatmap wiht correlations more than 0.7

# %%
threshold = 0.5
corr_matrix_trheshold = corr_matrix[abs(corr_matrix) >= threshold]
corr_matrix_trheshold = corr_matrix_trheshold[abs(corr_matrix) != 1.0]
corr_matrix_trheshold = corr_matrix_trheshold.dropna(axis=1, how='all')
corr_matrix_trheshold = corr_matrix_trheshold.dropna(axis=0, how='all')
fig, ax = plt.subplots(layout="constrained", figsize=(16, 8))
fig.suptitle(f"Correlations of attributes (> {threshold})", fontsize=26)

im = ax.imshow(corr_matrix_trheshold, aspect="auto")
ax.set_xticks(np.arange(len(corr_matrix_trheshold.columns)), labels=corr_matrix_trheshold.columns, fontsize=12)
ax.set_yticks(np.arange(len(corr_matrix_trheshold.index)), labels=corr_matrix_trheshold.index, fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(corr_matrix_trheshold.shape[0]):
    for j in range(corr_matrix_trheshold.shape[1]):
        text = ax.text(j, i, round(corr_matrix_trheshold.iloc[i, j], 2),
                       ha="center", va="center", color="w", fontsize=12,)

# %% [markdown]
# ### Select most valuabale features

# %%
corr_target = corr_matrix.loc[target]
corr_target = corr_target.drop("data_channel")
N = 3
corr_target_largest = corr_target.nlargest(N)
print(corr_target_largest)

# %% [markdown]
# ### Features Distribution

# %%
fig = plt.figure(constrained_layout=True, figsize=(20, 8))
subfigs = fig.subfigures(nrows=N, ncols=1)
for subfig, feature in zip(subfigs, corr_target_largest.index):
    subfig.suptitle(f"{feature}", fontsize=26)
    axes = subfig.subplots(nrows=1, ncols=2)
    df.hist(column=feature, ax=axes[0], color="orange", edgecolor="black", grid=False)
    axes[0].set_title("")
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[1].boxplot([df.query(f"{target} == \"{t}\"")[feature] for t in target_values])
    axes[1].set_xticks(np.arange(1, len(target_values) + 1), labels=target_values, fontsize=16)
plt.show()


# %% [markdown]
# ### Function to fit preconfigured model and score it

# %%
def fit_and_score(model):
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred, normalize="all")
    accuracies = map(lambda x: round(x, 2), np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    accuracy_table = pd.DataFrame({'Class': model.classes_, 'Accuracy': accuracies})
    table_data = accuracy_table.values

    fig, ax = plt.subplots(layout="constrained", figsize=(16, 8))
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                           display_labels=model.classes_)
    conf_matrix_plot.plot(ax=ax, colorbar=False)
    table_plot = ax.table(cellText=table_data,
                      colLabels=['Class', 'Accuracy'],
                      cellLoc='center', bbox=[0.1, -0.3, 0.8, 0.2])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    # print(conf_matrix)
    print(classification_report(y_test, y_pred))


# %% [markdown]
# ### Normalize data

# %%
scaler = MinMaxScaler()
features_norm = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# %% [markdown]
# ### Split dataset into train and test sets

# %%
X, y = features_norm, df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {y_train.value_counts()}")
print(f"Test: {y.value_counts()}")

# %% [markdown]
# ### KNN

# %%
# param_grid = {
#     "n_neighbors": np.arange(10, 25, 5),
#     "weights": ["uniform", "distance"],
#     "p": [1, 2],
#     "n_jobs": [-1],
# }
# grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=12, n_jobs=-1)
# grid.fit(X, y)
# print(grid.best_params_)
# fit_and_score(KNeighborsClassifier(**grid.best_params_))
best_params = {
        'n_jobs': -1, 
        'n_neighbors': 20,
        'p': 1,
        'weights': 'distance'
}
fit_and_score(KNeighborsClassifier(**best_params))


# %% [markdown]
# ### Decision Tree

# %%
param_grid = {
    "criterion" : ["gini", "entropy", "log_loss"],
    "class_weight" : [None, "balanced"],
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring="balanced_accuracy", cv=12, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
dtc = DecisionTreeClassifier(**grid.best_params_)
fit_and_score(dtc)

# %% [markdown]
# ### SVM linear

# %%
# {'C': 20, 'class_weight': 'balanced', 'decision_function_shape': 'ovo', 'kernel': 'linear'}
param_grid = {
    "C": np.arange(10, 25, 5),
    "kernel": ["linear"],
    "class_weight": [None, "balanced"],
    "decision_function_shape": ["ovo", "ovr"],
    "cache_size": [10_000],
}
grid = GridSearchCV(SVC(), param_grid, cv=12, n_jobs=-1, scoring="balanced_accuracy")
grid.fit(X, y)
print(grid.best_params_)
fit_and_score(SVC(**grid.best_params_))

# %% [markdown]
# ### SVM poly

# %%
# {'C': 20, 'cache_size': 10000, 'class_weight': 'balanced', 'decision_function_shape': 'ovo', 'degree': np.int64(2), 'gamma': 0.1, 'kernel': 'poly'}
param_grid = {
    "C": [1, 10, 20],
    "kernel": ["poly"],
    "degree": np.arange(2, 5, 1),
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "class_weight": ["balanced"],
    "decision_function_shape": ["ovo"],
    "cache_size": [10_000],
}
grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, scoring="balanced_accuracy")
grid.fit(X, y)
print(grid.best_params_)
fit_and_score(SVC(**grid.best_params_))

# %% [markdown]
# ### SVM rbf

# %%
param_grid = {
    "C": [1, 10, 20],
    "kernel": ["poly"],
    "degree": [2, 3, 6],
    "gamma": ["scale", "auto", 0.1, 1],
    "class_weight": ["balanced"],
    "decision_function_shape": ["ovo"],
    "cache_size": [10_000],
}
grid = GridSearchCV(SVC(), param_grid, cv=4, n_jobs=-1, scoring="balanced_accuracy")
grid.fit(X, y)
print(grid.best_params_)
fit_and_score(SVC(**grid.best_params_))

# %% [markdown]
# ### Random Forest

# %%
param_grid = {
    "n_estimators": [10, 100, 250, 300],
    "criterion" : ["gini", "entropy", "log_loss"],
    "class_weight" : ["balanced"],
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=4, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
fit_and_score(RandomForestClassifier(**grid.best_params_))

# %% [markdown]
# ### AdaBoost

# %%
param_grid = {
    "n_estimators": [10, 100, 250, 300],
    "learning_rate" : np.arange(1, 10, 3),
    "algorithm" : ["SAMME"],
}
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=12, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
fit_and_score(AdaBoostClassifier(**grid.best_params_))
