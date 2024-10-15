#!/usr/bin/env python

import pandas as pd
import numpy as np

# import data
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv", skipinitialspace=True)

# it's not fair to compare shares number for 8 and 30 days articles
df["shares_per_day"] = df["shares"] / df["timedelta"]

# drop not useful(for now) data 
df = df.drop(columns=["url", "timedelta", "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
                      "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", "weekday_is_sunday", 
                      "is_weekend", "shares"])

# choose target and features
targets = [
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
df["data_channel"] = df.apply(lambda row: [col for col in df.columns if row[col] and col.startswith("data_channel_")][0], axis=1)
df["data_channel"] = df["data_channel"].str.replace("data_channel_is_", "")
df = df.drop(columns=[col for col in df.columns if col.startswith("data_channel_is")])


df.info()
df.to_csv("OnlineNewsPopularity/OnlineNewsPopularity_classification_lab1.csv", index=False)

