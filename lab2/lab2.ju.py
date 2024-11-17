# %% [markdown]
# Lab Work #2
### Nedozhdii Oleksii FF-31mn

# %%
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"


# %%
df = pd.read_csv("resources/user_behavior_dataset.csv")

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ["Gender", "Device Model", "Operating System"]:
    df[col] = le.fit_transform(df[col])
target = "User Behavior Class"
features = df.columns.drop(labels=target)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_std = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
X_std, y = features_std, df[target]


# %% [markdown]
#### 1. Зниження розмірності і візуалізація даних

Застосуйте методи зниження розмірності sklearn.decomposition.PCA і sklearn.manifold.TSNE для візуалізації даних, з якими ви працювали в лабораторній № 1 (знижуючи розмірність до двох). Візуалізуйте результат.

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %%
pca = PCA()
X_pca = pca.fit_transform(X_std)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
x = list(range(0, len(exp_var_pca)))

# %%
fig = make_subplots(rows=2,cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
        xaxis2=dict(title="Principal component",
                    showgrid=True,))

step_trace = go.Scatter(x=x, y=cum_sum_eigenvalues,
                        name="Cumulative<br>explained variance",
                        mode="lines", line_shape="hv")
bar_trace = go.Bar(x=x, y=exp_var_pca,
                        name="Explained variance")

fig.add_trace(step_trace, row=1, col=1)
fig.add_trace(bar_trace, row=2, col=1)
iplot(fig)

# %%
df_pca = pd.DataFrame({"PCA 1": X_pca[:,0],
                       "PCA 2": X_pca[:,1],
                       target: y})
df_pca[target] = df_pca[target].astype("category")
fig = px.scatter(df_pca, x="PCA 1", y="PCA 2",
                 color=target)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
iplot(fig)

# %%
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_std)

# %%
df_tsne = pd.DataFrame({"t-SNE 1": X_tsne[:,0], "t-SNE 2": X_tsne[:,1], target: y})
df_tsne[target] = df_tsne[target].astype("category")
fig = px.scatter(df_tsne, x="t-SNE 1", y="t-SNE 2", color=target)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
iplot(fig)

# %% [markdown]
#### 2. Кластерний аналіз

За допомогою алгоритму k-means зробіть квантування зображення (видалення візуально надлишкової інформації) з глибиною 64, 32, 16 та 8 рівнів для будь-якого обраного самостійно зображення.

# %%
from sklearn.cluster import KMeans
from skimage import io

# %%
image = io.imread("resources/pic2.jpg")

# %%
def quantize_image(image, depth: int):
    image = np.array(image, dtype=np.float64) / 255
    w, h, d = image.shape
    image_array = image.reshape((w * h, d))
    kmeans = KMeans(n_clusters=depth, random_state=42).fit(image_array)
    labels = kmeans.predict(image_array)
    quantized_image = kmeans.cluster_centers_[labels].reshape(w, h, d)
    fig = trace_quantized = px.imshow(quantized_image)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_xaxes(title=f"{depth} quantize level", showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    iplot(fig)

# %%
quantize_image(image, 8)
# %%
quantize_image(image, 16)
# %%
quantize_image(image, 32)
# %%
quantize_image(image, 64)

# %% [markdown]
#### 3. Обробка та класифікація текстових даних

Завантажте набір текстових даних (з мітками класів). Проведіть передобробку даних (видаліть стоп-слова, пунктуацію), за допомогою wordcloud зробіть візуалізацію найбільш поширених слів або n-gram у кожному класі. Векторизуйте тексти (наприклад за допомогою sklearn.feature_extraction.text.TfidfVectorizer). Проведіть класифікацію текстових даних, зробіть оцінку якості.

# %% [markdown]
**Dataset**

https://archive.ics.uci.edu/dataset/456/multimodal+damage+identification+for+humanitarian+computing

5879 text from social media related to damage during natural disasters/wars, and belong to 6 classes: Fires, Floods, Natural landscape, Infrastructural, Human, Non-damage.


# %%
import os
from itertools import chain

# %%
path = "resources/multimodal"
classes = os.listdir(path)
print(classes)
text_pathes = []
for class_dir in classes:
    class_path = os.path.join(path, class_dir)
    text_pathes.append(
            [os.path.join(class_path, file_name) for file_name in os.listdir(class_path)]
    )
class_column = []
for class_, text_pathes_per_class  in zip(classes, text_pathes):
    class_column.append([class_] * len(text_pathes_per_class))
class_column = list(chain.from_iterable(class_column))
text_path_column = list(chain.from_iterable(text_pathes))

df = pd.DataFrame({"file_path": text_path_column, "class": class_column})

# %%
import string
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %%
nltk.download("stopwords")
nltk.download("punkt_tab")

# %%
text_column = []
for file_path in df["file_path"]:
    with open(file_path) as f: text_column.append(f.read())
df["raw_text"] = text_column

# %%
formated_text_column = []
stop_words = stopwords.words("english")
for raw_text in df["raw_text"]:
    formated_text = raw_text.lower()
    formated_text = re.sub(r"@\S+", " ", formated_text) # remove mentions
    formated_text = re.sub(r"https*\S+", " ", formated_text) # remove url
    formated_text = re.sub(r"#\S+", " ", formated_text) # remove hashtags
    formated_text = re.sub(r"\d", " ", formated_text) # remove all numbers
    formated_text = re.sub(r"[%s]" % re.escape(string.punctuation), ' ', formated_text) # remove punctuations
    formated_text = re.sub(r"\s{2,}", " ", formated_text) # remove extra spaces
    formated_text = re.sub(r'[^\x00-\x7F]+', '', formated_text) # remove not ascii
    formated_text = formated_text.strip()
    formated_text = ' '.join([word for word in formated_text.split(' ') if word not in stop_words])
    formated_text_column.append(formated_text)
df["formated_text"] = formated_text_column
df = df.query("`formated_text` != ''").reset_index(drop=True)

# %%
i = np.random.randint(len(df))
print("=============File path==============")
print(df.loc[i, "file_path"], end="\n\n")
print("=============Raw text===============")
print(df.loc[i, "raw_text"], end="\n\n")
print("=============Formated text==========")
print(df.loc[i, "formated_text"], end="\n\n")

# %%
concated_text_per_class = {}
for class_ in df["class"].unique():
    text = ""
    for t in df.query(f"`class` == '{class_}'")["formated_text"]:
        text += t
    concated_text_per_class[class_] = text

# %%
from wordcloud import WordCloud

# %%
for i, class_ in enumerate(df["class"].unique()):
    wordcloud = WordCloud(
        # background_color="#F9F9FA",
        collocations=True,
        max_words=30,
    ).generate(concated_text_per_class[class_])
    fig = px.imshow(wordcloud)
    fig.update_xaxes(showticklabels=False, showgrid=False, automargin=True)
    fig.update_yaxes(showticklabels=False, showgrid=False, automargin=True)
    fig.update_layout(autosize=True, title=class_)
    iplot(fig)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
### Naive Bayes Classifier

For text classsictaion Multinomial NB or Complement NB
- dataset balanced →  Multinomial NB
- dataset unbalanced → Complement NB

# %%
fig = px.histogram(df, x="class", title="Unbalanced")
iplot(fig)


# %%
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix

# %%
encoder = LabelEncoder()
df["class_num"] = encoder.fit_transform(df["class"])

# %%
X, y = df["formated_text"], df["class_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"{X_train.shape=}")
print(f"{X_test.shape=}")

# %%
tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=5)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# %%
count_vectorizer = CountVectorizer(min_df=5)
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# %%
clf_tfidf = ComplementNB(alpha=0.6)
clf_tfidf.fit(tfidf_train, y_train)


# %%
clf_count = ComplementNB(alpha=0.6)
clf_count.fit(count_train, y_train)

# %%
y_pred_tfidf = clf_tfidf.predict(tfidf_test)
print(classification_report(y_test, y_pred_tfidf))

# %%
conf_matrix = confusion_matrix(y_test, y_pred_tfidf, normalize="all")
print(conf_matrix)

# %%
y_pred_count = clf_count.predict(tfidf_test)
print(classification_report(y_test, y_pred_count))

# %%
conf_matrix = confusion_matrix(y_test, y_pred_count, normalize="all")
print(conf_matrix)


# %% [markdown]
- TFIDF accuracy: 0.69
- Count accuraycy: 0.7
