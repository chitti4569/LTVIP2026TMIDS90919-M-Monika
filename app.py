import numpy as np  # linear algebra
import pandas as pd  # data processing
import joblib
pd.set_option('display.max_columns', None)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

data = pd.read_csv('data/startup_data.1.csv')

data.head()

data['State'] = 'other'
data.loc[data['state_code'] == 'CA', 'State'] = 'CA'
data.loc[data['state_code'] == 'NY', 'State'] = 'NY'
data.loc[data['state_code'] == 'MA', 'State'] = 'MA'
data.loc[data['state_code'] == 'TX', 'State'] = 'TX'
data.loc[data['state_code'] == 'WA', 'State'] = 'WA'

state_count = data['State'].value_counts()
plt.pie(state_count, labels=state_count.index, autopct='%1.1f%%')
plt.show()

data['category'] = 'other'
data.loc[data['category_code'] == 'software', 'category'] = 'software'
data.loc[data['category_code'] == 'web', 'category'] = 'web'
data.loc[data['category_code'] == 'mobile', 'category'] = 'mobile'
data.loc[data['category_code'] == 'enterprise', 'category'] = 'enterprise'
data.loc[data['category_code'] == 'advertising', 'category'] = 'advertising'
data.loc[data['category_code'] == 'games_video', 'category'] = 'games_video'
data.loc[data['category_code'] == 'semiconductor', 'category'] = 'semiconductor'
data.loc[data['category_code'] == 'network_hosting', 'category'] = 'network_hosting'
data.loc[data['category_code'] == 'biotech', 'category'] = 'biotech'
data.loc[data['category_code'] == 'hardware', 'category'] = 'hardware'

category_count = data['category'].value_counts()
plt.pie(category_count, labels=category_count.index, autopct='%1.1f%%')
plt.show()

prop_df = data.groupby('status').size().reset_index(name='counts')
prop_df['proportions'] = prop_df['counts'] / prop_df['counts'].sum()

sns.barplot(data=prop_df, x='status', y='proportions')
plt.title('Distribution of Status of the Startup')
plt.show()

fig, ax = plt.subplots(figsize=(30, 10))

prop_df = (
    data.groupby(['category', 'status'])
        .size()
        .reset_index(name='counts')
)

# compute proportions correctly
prop_df['proportions'] = (
    prop_df['counts'] /
    prop_df.groupby('category')['counts'].transform('sum')
)

sns.barplot(
    data=prop_df,
    x='category',
    y='proportions',
    hue='status',
    ax=ax
)

plt.show()

# 1. Clean the column names to remove hidden spaces
data.columns = data.columns.str.strip()

# 2. Check if the column exists before plotting
if 'founded_year' in data.columns and 'category' in data.columns:
    cat_year = pd.crosstab(index=data['founded_year'], columns=data['category'])

    fig, ax = plt.subplots(figsize=(20, 8))  # 20x8 is usually plenty for readability
    sns.lineplot(data=cat_year, lw=3)

    plt.title('Category Wise Evolution of Startups')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Keeps legend from blocking the lines
    plt.tight_layout()
    plt.show()
else:
    print("Error: One of the columns is missing. Available columns are:")
    print(data.columns.tolist())


plt.show()


print(data.columns.tolist())

# Create founded_year correctly
data['founded_at'] = pd.to_datetime(data['founded_at'], errors='coerce')
data['founded_year'] = data['founded_at'].dt.year

# Plot
sns.catplot(
    data=data,
    x="founded_year",
    y="funding_total_usd",
    kind="box",
    height=5,
    aspect=2
)

plt.xticks(rotation=90)
plt.show()


fig, ax = plt.subplots(figsize=(20, 10))

d = data.loc[data['status'] == '1']
f = d[["has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD"]]

sns.countplot(data=pd.melt(f), x='variable', hue='value')
plt.show()


data.describe(include=['float64', 'int64'])
plt.show()


fig, ax = plt.subplots(figsize=(30, 10))

corr = data.select_dtypes(include=['int64', 'float64']).corr()

sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.show()


print(data['state_code'].equals(data['state_code.1']))
False
df = data.copy()
df['state_match'] = data['state_code'] == data['state_code.1']
print(df[['state_code', 'state_code.1', 'state_match']])
state = data['state_code'].value_counts().to_frame(name='count')
state['proportion'] = state['count'] / state['count'].sum() * 100
state
plt.show()

# Define features (X) and target (y)
X = data.drop(columns=['status'])
y = data['status']

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
plt.show()