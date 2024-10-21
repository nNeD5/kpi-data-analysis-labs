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
    plt.setp(ax.get_xtcklabels(), rotation=45, ha="right", rotation_mode="anchor")
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


def fit_and_score(model):
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

    print(conf_matrix)
    print(classification_report(y_test, y_pred))

# plot_heatmap()
# plot_corr_heatmap()
# plot_feature_distribution()

# kNN
param_grid = {
    "n_neighbors": [1, 5, 10, 15],
    "weights": ["uniform", "distance"],
    "p": [1, 2, 3],
    "n_jobs": [-1],
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
fit_and_score(KNeighborsClassifier(**grid.best_params_))


# Decision Treee
# param_grid = {
#     "criterion": ["gini","entropy"],
#     "max_depth": np.arange(3, 15)
# }
# grid = GridSearchCV(DecisionTreeClassifier(), param_grid)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# fit_and_score(DecisionTreeClassifier(**grid.best_params_))
