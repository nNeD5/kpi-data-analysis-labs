#!/usr/bin/env python


"""
Lab 1
-----
1) [x]  Завантажити дані, вивести назви колонок і розмір датасета
2) [x] Опрацювати пропуски (по можливості заповнити їх або видалити)
3) [x] Візуалізувати дані: побудувати графік (heatmap), що відображає кореляції
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
import matplotlib.pyplot as plt


plt.style.use("default")


# import data
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv", skipinitialspace=True)

# it's not fair to compare shares number for 8 and 30 days articles
df["shares_per_day"] = df["shares"] / df["timedelta"]

# check and drop null valus
null_indexes = sum(np.nonzero(pd.isnull(df)))
if len(null_indexes) > 0:
    print("Ups, spot null value. Going to drop it.")
    df = df.dropna()

# discover and remove anomaly
df_old = df.copy()
df = df.query("shares_per_day < 75")

def plot_anomaly():
    fig, axes = plt.subplots(3, layout="constrained")
    fig.suptitle("Shares per day distribution", fontsize=26)
    axes[0].hist(df_old.shares_per_day, bins=100, color="orange", edgecolor="black")
    axes[0].text(1000, 20000, f"{df.shape[0]} values numer",  color="black", fontsize=16)

    df_200 = df_old.query("shares_per_day < 200")
    axes[1].hist(df_200.shares_per_day, bins=100, edgecolor="black")
    axes[1].axvline(x=75, color="red", linestyle="--", linewidth=4)
    axes[1].text(90, 3000, f"Anomaly",  color="red", fontsize=32)
    axes[1].text(90, 2000, f"{df.shape[0]} values numer",  color="black", fontsize=16)
    axes[1].set_xlim(0, 200)

    df_75 = df_old.query("shares_per_day < 75")
    axes[2].hist(df.query("shares_per_day < 75").shares_per_day, bins=100, color="skyblue", edgecolor="black")
    axes[2].set_xlim(0, 75)
    axes[2].text(40, 2000, "Data in use",  color="black", fontsize=32)
    axes[2].text(40, 1000, f"{df.shape[0]} values numer",  color="black", fontsize=16)

    plt.show()


# present data
print(df.head())        
print(df.columns)
correlation_matrix = df.corr(numeric_only=True)
print("Correlation matrix:")
print(correlation_matrix["shares_per_day"][["num_imgs", "num_videos", "n_tokens_content", "n_tokens_title"]])
print("-----------------------------------\n")

# min-Max normalization
df_numeric = df.select_dtypes(include='number')
df_min_max_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
df_norm= df.copy()
df_norm[df_numeric.columns] = df_min_max_normalized

# select data for traning
df_train = df_norm.sample(frac=0.8, random_state=42)


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
    features = ["n_tokens_content", "n_tokens_title", "num_imgs", "num_videos"]
    print(df[features])
    x_labels = ["Words in content", "Words in title", "Number of images", "Number of videos"]
    ranges = [(0, 2700), (2, 15), (0, 30), (0, 10)]
    bins_number=50

    fig, axes = plt.subplots(4, 2, layout="constrained")
    fig.suptitle("Features distibution and popularity dependens on feature", fontsize=26)
    for ax, range_, x_label, feature in zip(axes, ranges, x_labels, features):
        ax[0].hist(df[feature], bins=bins_number,color="orange", edgecolor="black", range=range_)
        ax[0].set_ylabel("Frequency", fontsize=16)
        ax[0].set_xlabel(x_label, fontsize=16)
        shares, bins = np.histogram(df[feature], bins=bins_number, weights=df["shares_per_day"] / df[feature].shape[0], range=range_)
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
        ax[1].bar(bin_centers, shares, width=range_[1] / bins_number, color="skyblue", edgecolor="black")
        ax[1].set_ylabel("Shares per article in day", fontsize=16)
        ax[1].set_xlabel(x_label, fontsize=16)
    axes[0][1].annotate("Perhaps due to\nonly video/img aticles", 
            xy=(11, 0.48),
            xytext=(1500, 0.753),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=16,
            color="red",)
    plt.show()


def plot_box():
    fig, ax = plt.subplots(layout="constrained")
    fig.suptitle("Words in title", fontsize=26)
    ax.boxplot(df.n_tokens_title)
    plt.show()


def knn():
    features = ["n_tokens_content", "n_tokens_title", "num_imgs", "num_videos"]
    target = "shares_per_day"

    # n=1: 0.37077098388735175 -> best
    # n=5: 0.1872053176926296
    # for n in range(1, 30):
    #     model = KNeighborsRegressor(n_neighbors=n, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    #     model.fit(X=df_train[features], y=df_train[target])
    #     print(n, model.score(df_norm[features], df_norm[target]))
    
    targets = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
    ]
    featrues = ["n_tokens_title", "title_subjectivity", "title_subjectivity", "title_sentiment_polarity"]
    model = KNeighborsClassifier(n_neighbors=2, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    # for n in range(1, 30):
    #     model = KNeighborsClassifier(n_neighbors=n, weights="uniform", algorithm="brute", metric="euclidean", n_jobs=-1)
    #     model.fit(X=df_train[features], y=df_train[target])
    #     print(n, model.score(df_norm[features], df_norm[target]))
    # fig, ax = plt.subplots(layout="constrained")
    # plt.show()

# plot_anomaly()
# plot_heatmap()
# plot_histogram()
# plot_box()
knn()
