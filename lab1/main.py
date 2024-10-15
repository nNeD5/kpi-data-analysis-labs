#!/usr/bin/env python


"""
Lab 1
-----
1) [x]  Завантажити дані, вивести назви колонок і розмір датасета
2) [x] Опрацювати пропуски (по можливості заповнити їх або видалити)
3) [ ] Візуалізувати дані: побудувати графік (heatmap), що відображає кореляції
ознак між собою і з цільовою змінною (розміткою); побудувати гістограми
розподілу ознак і boxplot-и ознак відносно цільової змінної (якщо ознак занадто багато
обмежитися декількома)
4) [x] Нормалізувати дані
5) [ ] Провести навчання наступних класифікаторів:
- [ ] kNN
- [ ] дерево ухвалення рішень
- [ ] SVM
- [ ] Random Forest
- [ ] AdaBoost
Підібрати оптимальні параметри
- для kNN
- для SVM за допомогою GridSearch підібрати оптимальні `C` і `gamma`
- [ ] Серед обраних оптимальних моделей кожного класу вибрати найкращу.
- [ ] Відобразити
sklearn.metrics.classification_report і sklearn.metrics.confusion_matrix
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


plt.style.use("default")


# import data
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity_classification_lab1.csv", skipinitialspace=True)
target = "data_channel"
features = df.columns.drop(labels=target)

# present data
df.info()

# min-max normalization
df_numeric = df.select_dtypes(include='number')
df_min_max_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
df_norm= df.copy()
df_norm[df_numeric.columns] = df_min_max_normalized

# split dataset into train and test sets
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# plot: heatmap
def plot_heatmap():
    channels = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]
    days = [
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
    ]
    shares = []
    for day in days:
        shares_at_day = []
        for channel in channels:
            df_at_day_in_channel = df.query(f"{channel} == True & {day} == True")
            shares_at_day.append(sum(df_at_day_in_channel.shares_per_day / df_at_day_in_channel.shape[0]))
        shares.append(shares_at_day)
    shares = np.array(shares)

    channels_labels = [
        "Lifestyle",
        "Entertainment",
        "Business",
        "Social Media",
        "Tech",
        "World",
    ]
    days_labels = [day.split("_")[2] for day in days]

    _, ax = plt.subplots()
    im = ax.imshow(shares, aspect="auto")
    ax.set_xticks(np.arange(len(channels_labels)), labels=channels_labels, fontsize=16)
    ax.set_yticks(np.arange(len(days_labels)), labels=days_labels, fontsize=16)
    ax.set_title("Distribuion of shares per article in day", fontsize=26)
    for i in range(len(days)):
        for j in range(len(channels)):
            text = ax.text(j, i, round(shares[i, j], 2),
                           ha="center", va="center", color="w")
    plt.show()

# plot: histogram
def plot_histogram():
    targets = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]
    labels = ["Lifestyle", "Entertainment", "Business", "Social Media", "Tech", "World"]
    fig, ax = plt.subplots(layout="constrained")
    article_number = [len(df.query(f"{target} == True")) for target in targets]
    ax.bar(labels, article_number, color="skyblue", edgecolor="black")
    ax.set_ylabel("Number of articles", fontsize=16)
    plt.show()

# plot: box
def plot_box():
    fig, ax = plt.subplots(layout="constrained")
    fig.suptitle("Words in title", fontsize=26)
    ax.boxplot(df.n_tokens_title)
    plt.show()


def knn():
    # n=1: score=0.9375396489744132
    # n 2: score=0.7086064707126243
    # n 3: score=0.795146965531825
    # n 4: score=0.7212677098752379
    # n 5: score=0.7679213364347642
    # n 6: score=0.7232765912455065

    model = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
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

# plot_anomaly()
# plot_heatmap()
# plot_histogram()
# plot_box()
knn()
