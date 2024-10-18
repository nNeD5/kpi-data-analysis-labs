#!/usr/bin/env python


"""
Lab 1
-----
1) [x] Завантажити дані, вивести назви колонок і розмір датасета
2) [x] Опрацювати пропуски (по можливості заповнити їх або видалити)
3) [x] Візуалізувати дані: побудувати графік (heatmap), що відображає кореляції
ознак між собою і з цільовою змінною (розміткою); побудувати гістограми
розподілу ознак і boxplot-и ознак відносно цільової змінної (якщо ознак занадто багато
обмежитися декількома)
4) [x] Нормалізувати дані
5) [ ] Провести навчання наступних класифікаторів:
- [x] kNN
- [.] дерево ухвалення рішень
- [.] SVM
- [.] Random Forest
- [.] AdaBoost
Підібрати оптимальні параметри
- [ ] для kNN
- [ ] для SVM за допомогою GridSearch підібрати оптимальні `C` і `gamma`
- [ ] Серед обраних оптимальних моделей кожного класу вибрати найкращу.
- [ ] Відобразити
sklearn.metrics.classification_report і sklearn.metrics.confusion_matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV


plt.style.use("default")


# import data
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity_classification_lab1.csv", skipinitialspace=True)
target = "data_channel"
features = df.columns.drop(labels=target)
target_values = df["data_channel"].unique()
features_number = len(features)
target_number = len(target_values)

# present data
df.info()

# min-max normalization
df_numeric = df.select_dtypes(include='number')
df_min_max_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
df_norm = df.copy()
df_norm[df_numeric.columns] = df_min_max_normalized

# split dataset into train and test sets
X, y = df_norm[features], df_norm[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# plot: heatmap
def plot_heatmap():
    fig, axes = plt.subplots(3, 3, layout="constrained")
    fig.suptitle("Average feature values", fontsize=26)
    for ax_i, ax in enumerate(axes.flatten()):
        features_slice_number = int(len(features) / 9)
        features_slise = features[ax_i * features_slice_number: (ax_i + 1) * features_slice_number]
        average_feature_value = []
        for feature in features_slise:
            average_feature_per_channel = []
            for target_value in target_values:
                df_channel = df_norm.query(f"data_channel == \"{target_value}\"")
                average_feature_per_channel.append(df_channel[feature].sum() / len(df_channel))
            average_feature_value.append(average_feature_per_channel)
        average_feature_value = np.array(average_feature_value)

        im = ax.imshow(average_feature_value, aspect="auto")
        ax.set_xticks(np.arange(target_number), labels=target_values, fontsize=16)
        ax.set_yticks(np.arange(features_slice_number), labels=features_slise, fontsize=16)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(features_slice_number):
            for j in range(target_number):
                text = ax.text(j, i, round(average_feature_value[i, j], 2),
                               ha="center", va="center", color="w")
    plt.show()

def plot_corr_heatmap():
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[target] = le.fit_transform(df[target])

    corr_matrix = df_encoded.corr()
    print(corr_matrix)

    fig, ax = plt.subplots(layout="constrained")
    fig.suptitle("Correlations of attributes", fontsize=26)

    im = ax.imshow(corr_matrix, aspect="auto")
    ax.set_xticks(np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, fontsize=16)
    ax.set_yticks(np.arange(len(corr_matrix.index)), labels=corr_matrix.index, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                           ha="center", va="center", color="w")
    plt.show()

    threshold = 0.7
    corr_matrix = corr_matrix[abs(corr_matrix) >= threshold]
    corr_matrix = corr_matrix[abs(corr_matrix) != 1.0]
    corr_matrix = corr_matrix.dropna(axis=1, how='all')
    corr_matrix = corr_matrix.dropna(axis=0, how='all')
    fig, ax = plt.subplots(layout="constrained")
    fig.suptitle(f"Correlations of attributes (> {threshold})", fontsize=26)

    im = ax.imshow(corr_matrix, aspect="auto")
    ax.set_xticks(np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, fontsize=16)
    ax.set_yticks(np.arange(len(corr_matrix.index)), labels=corr_matrix.index, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                           ha="center", va="center", color="w")
    plt.show()

# plot: histogram
def plot_feature_distribution():
    # choose features for hist
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[target] = le.fit_transform(df[target])
    corr_matrix = df_encoded.corr()
    corr_target = corr_matrix.loc[target]
    corr_target = corr_target.drop("data_channel")
    corr_target = corr_target[~corr_target.index.str.startswith("LDA")]
    N = 3
    corr_target_largest = corr_target.nlargest(N)
    print(corr_target_largest)

    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=N, ncols=1)
    for subfig, feature in zip(subfigs, corr_target_largest.index):
        subfig.suptitle(f"{feature}", fontsize=26)
        axes = subfig.subplots(nrows=1, ncols=2)
        df.hist(column=feature, ax=axes[0], color="orange", edgecolor="black", grid=False)
        axes[0].set_title("")
        axes[0].set_ylabel("Frequency", fontsize=16)
        axes[1].boxplot([df.query(f"{target} == \"{t}\"")[feature] for t in target_values])
        axes[1].set_xticks(np.arange(1, target_number + 1), labels=target_values, fontsize=16)
    plt.show()

def knn():
    # n=1: score=0.9375396489744132
    # n 2: score=0.7086064707126243
    # n 3: score=0.795146965531825
    # n 4: score=0.7212677098752379
    # n 5: score=0.7679213364347642
    # n 6: score=0.7232765912455065

    # for n in range(1, 6):
    #     model = KNeighborsClassifier(n_neighbors=n, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    #     model.fit(X=X_train, y=y_train)
    #     print(n, model.score(X_test, y_test))


    model = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred, normilize="all")
    accuracies = map(lambda x: round(x, 2), np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    accuracy_table = pd.DataFrame({'Class': model.classes_, 'Accuracy': accuracies})
    table_data = accuracy_table.values

    fig, ax = plt.subplots(layout="constrained")
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                           display_labels=model.classes_)
    conf_matrix_plot.plot(ax=ax, colorbar=False)
    table_plot = ax.table(cellText=table_data,
                      colLabels=['Class', 'Accuracy'],
                      cellLoc='center', bbox=[0.1, -0.25, 0.8, 0.2])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(16)
    plt.show()

def svm():
    model = SVC()
    parameters = {
                "kernel": ["poly", "rbf", "sigmoid",],
                "C": [i for i in range(1, 10)],
                "gamma": [i for i in range(1, 10)],
            }
    model = GridSearchCV(model, parameters, n_jobs=-1)
    model.fit(X_train, y_train)
    print(model.best_params_)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred, normalize="all")
    accuracies = map(lambda x: round(x, 2), np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    accuracy_table = pd.DataFrame({'Class': model.classes_, 'Accuracy': accuracies})
    table_data = accuracy_table.values

    fig, ax = plt.subplots(layout="constrained")
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                           display_labels=model.classes_)
    conf_matrix_plot.plot(ax=ax, colorbar=False)
    table_plot = ax.table(cellText=table_data,
                      colLabels=['Class', 'Accuracy'],
                      cellLoc='center', bbox=[0.1, -0.25, 0.8, 0.2])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(16)
    plt.show()

def decision_tree():
    model = DecisionTreeClassifier()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred, normalize="all")
    accuracies = map(lambda x: round(x, 2), np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    accuracy_table = pd.DataFrame({'Class': model.classes_, 'Accuracy': accuracies})
    table_data = accuracy_table.values

    fig, ax = plt.subplots(layout="constrained")
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                           display_labels=model.classes_)
    conf_matrix_plot.plot(ax=ax, colorbar=False)
    table_plot = ax.table(cellText=table_data,
                      colLabels=['Class', 'Accuracy'],
                      cellLoc='center', bbox=[0.1, -0.25, 0.8, 0.2])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(16)
    plt.show()

def fit(model):
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred, normalize="all")
    accuracies = map(lambda x: round(x, 2), np.diag(conf_matrix) / np.sum(conf_matrix, axis=1))
    accuracy_table = pd.DataFrame({'Class': model.classes_, 'Accuracy': accuracies})
    table_data = accuracy_table.values

    fig, ax = plt.subplots(layout="constrained")
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                           display_labels=model.classes_)
    conf_matrix_plot.plot(ax=ax, colorbar=False)
    table_plot = ax.table(cellText=table_data,
                      colLabels=['Class', 'Accuracy'],
                      cellLoc='center', bbox=[0.1, -0.25, 0.8, 0.2])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(16)
    print(classification_report(y_test, y_pred))
    plt.show()

# plot_heatmap()
# plot_corr_heatmap()
# plot_feature_distribution()
# knn()
svm()
# decision_tree()
# fit(SVC())
