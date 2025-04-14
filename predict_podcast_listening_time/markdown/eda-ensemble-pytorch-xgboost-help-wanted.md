# Predicting Podcast Listening Times using PyTorch

## Competition Summary 📘
Welcome to the 2025 Kaggle Playground Series!

This series continues the tradition of offering fun, approachable competitions for the community to sharpen their machine learning skills. Each month, a new challenge arrives — and this time, your mission is to predict the listening time of a podcast episode.

The dataset (both train and test) was synthetically generated using a deep learning model trained on the Podcast Listening Time Prediction dataset. While the feature distributions are quite similar to the original, they’re not identical — feel free to explore the differences or even incorporate the original dataset into your workflow if it helps improve model performance.

📏 Evaluation Metric: Root Mean Squared Error (RMSE)

## My Approach 🧠 
My plan for this competition is fairly straightforward:

Exploratory Data Analysis (EDA) – to understand patterns, distributions, and spot any quirks in the data.

Modeling:

- Set up a data preprocessing pipeline.

- Train both a PyTorch neural network and an XGBoost model.

- Finally, ensemble the two models for improved predictions.

I'm using PyTorch mainly for learning purposes. That said, XGBoost is currently outperforming my NN, both in speed and accuracy. Still, I’m keen to keep experimenting and improving my neural network setup.

💡 If you have tips on optimizing PyTorch models for tabular data, or suggestions for handling embeddings better — I’d love to hear them!

## Current Issues & Help Wanted ⚠️ 
I'm running into a problem with the embedding layer in my neural network. When I generate new categorical features (e.g., via combinations of existing columns), some rare values show up in the validation/test set that weren’t seen during training. These unseen values are causing the embedding to crash.

I tried using the `OrdinalEncoder` with `handle_unknown='use_encoded_value'` and encoding unseen values as `-1`, then passing `padding_idx=1` to the embedding layer (as ChatGPT suggested) — but it’s still not working as expected.

Any help or ideas on how to safely handle unseen categorical values in embeddings would be greatly appreciated!


**Lets get started!!!!**


```python
# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import warnings

# Disable warning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option("display.max_columns", 500)
```


```python
# Load training and test data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

print(f"Shape of df_train: {df_train.shape}, Shape of df_test: {df_test.shape}")

df_test.head()
```

    Shape of df_train: (750000, 12), Shape of df_test: (250000, 11)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Podcast_Name</th>
      <th>Episode_Title</th>
      <th>Episode_Length_minutes</th>
      <th>Genre</th>
      <th>Host_Popularity_percentage</th>
      <th>Publication_Day</th>
      <th>Publication_Time</th>
      <th>Guest_Popularity_percentage</th>
      <th>Number_of_Ads</th>
      <th>Episode_Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>750000</td>
      <td>Educational Nuggets</td>
      <td>Episode 73</td>
      <td>78.96</td>
      <td>Education</td>
      <td>38.11</td>
      <td>Saturday</td>
      <td>Evening</td>
      <td>53.33</td>
      <td>1.0</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>750001</td>
      <td>Sound Waves</td>
      <td>Episode 23</td>
      <td>27.87</td>
      <td>Music</td>
      <td>71.29</td>
      <td>Sunday</td>
      <td>Morning</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>750002</td>
      <td>Joke Junction</td>
      <td>Episode 11</td>
      <td>69.10</td>
      <td>Comedy</td>
      <td>67.89</td>
      <td>Friday</td>
      <td>Evening</td>
      <td>97.51</td>
      <td>0.0</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>750003</td>
      <td>Comedy Corner</td>
      <td>Episode 73</td>
      <td>115.39</td>
      <td>Comedy</td>
      <td>23.40</td>
      <td>Sunday</td>
      <td>Morning</td>
      <td>51.75</td>
      <td>2.0</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>750004</td>
      <td>Life Lessons</td>
      <td>Episode 50</td>
      <td>72.32</td>
      <td>Lifestyle</td>
      <td>58.10</td>
      <td>Wednesday</td>
      <td>Morning</td>
      <td>11.30</td>
      <td>2.0</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>



# Exploratory data analysis

## Missing values

When examining the dataset, only a few features contain substantial missing values:

- `Episode_Length_minutes`

- `Guest_Popularity_percentage`

Both of these have missing values in both the training and test sets, making them important to handle thoughtfully.

There’s also a minor issue with:

- `Number_of_Ads`
This feature only has one missing value in the training set, so it shouldn't be too difficult to deal with.


```python
print("Missing values in the training data:")
print(df_train.isna().sum())
print("\nMissing values in the test data:")
print(df_test.isna().sum())
```

    Missing values in the training data:
    id                                  0
    Podcast_Name                        0
    Episode_Title                       0
    Episode_Length_minutes          87093
    Genre                               0
    Host_Popularity_percentage          0
    Publication_Day                     0
    Publication_Time                    0
    Guest_Popularity_percentage    146030
    Number_of_Ads                       1
    Episode_Sentiment                   0
    Listening_Time_minutes              0
    dtype: int64
    
    Missing values in the test data:
    id                                 0
    Podcast_Name                       0
    Episode_Title                      0
    Episode_Length_minutes         28736
    Genre                              0
    Host_Popularity_percentage         0
    Publication_Day                    0
    Publication_Time                   0
    Guest_Popularity_percentage    48832
    Number_of_Ads                      0
    Episode_Sentiment                  0
    dtype: int64
    

## Episode length ⏱️

As expected, `Episode_Length_minutes` shows a strong positive correlation with the target variable, `Listening_Time_minutes`. That makes sense — longer episodes generally offer more listening time.

However, this feature comes with a few quirks:

There are some extreme outliers, most notably an episode in the test set listed as 78,486,264.0 minutes long (!).

Additionally, there are instances where `Listening_Time_minutes` exceeds the episode length, which shouldn't be possible under normal circumstances.

These inconsistencies are likely due to the synthetic nature of the dataset, as mentioned in various discussion posts on the forum.


```python
print(f"The minimum episode length in the training data is {min(df_train['Episode_Length_minutes'].dropna())} minutes")
print(f"The maximum episode length in the training data is {max(df_train['Episode_Length_minutes'].dropna())} minutes")
print(f"The training data contains {(df_train['Episode_Length_minutes'] < 2).sum()} rows with episodes lengths < 2 minutes")
print(f"The training data contains {(df_train['Episode_Length_minutes'] > 120).sum()} rows with episodes lengths > 120 minutes")

print(f"\nThe minimum episode length in the test data is {min(df_test['Episode_Length_minutes'])} minutes")
print(f"The maximum episode length in the test data is {max(df_test['Episode_Length_minutes'])} minutes")
print(f"The test data contains {(df_test['Episode_Length_minutes'] < 2).sum()} rows with episodes lengths < 2 minutes")
print(f"The test data contains {(df_test['Episode_Length_minutes'] > 120).sum()} rows with episodes lengths > 120 minutes")
```

    The minimum episode length in the training data is 0.0 minutes
    The maximum episode length in the training data is 325.24 minutes
    The training data contains 4 rows with episodes lengths < 2 minutes
    The training data contains 9 rows with episodes lengths > 120 minutes
    
    The minimum episode length in the test data is 2.47 minutes
    The maximum episode length in the test data is 78486264.0 minutes
    The test data contains 0 rows with episodes lengths < 2 minutes
    The test data contains 4 rows with episodes lengths > 120 minutes
    


```python
# Set up the figure with 2 subplots (side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Set the style and color palette
sns.set(style="white", palette="husl")

bins = 100

# Cap episode length values at 150 for easy plotting
df_train["Episode_Length_minutes"] = df_train["Episode_Length_minutes"].apply(lambda x: min(x, 120))
df_test["Episode_Length_minutes"] = df_test["Episode_Length_minutes"].apply(lambda x: min(x, 120))

# 1. Histogram of Episode Length 
sns.histplot(df_train["Episode_Length_minutes"], bins=bins, stat='percent', label='Train data', color=sns.color_palette("husl")[0], ax=axes[0])
sns.histplot(df_test["Episode_Length_minutes"], bins=bins, stat='percent', label='Test data', color=sns.color_palette("husl")[1], ax=axes[0])
axes[0].set_title("Histogram of Episode Length (minutes)")
axes[0].set_xlabel("Episode Length (minutes)")
axes[0].set_ylabel("Percentage")
axes[0].legend(loc='upper left')
axes[0].set_xlim(0, 125)

# 2. Scatter plot of Listening Time vs Episode Length 
sns.scatterplot(data=df_train, x="Episode_Length_minutes", y="Listening_Time_minutes", color=sns.color_palette("husl")[0], ax=axes[1])
axes[1].plot([0, 120],[0, 120],
             color="black", linestyle="--", linewidth=1)
axes[1].set_title("Listening Time vs Episode Length")
axes[1].set_xlabel("Episode Length (minutes)")
axes[1].set_ylabel("Listening Time (minutes)")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_12_0.png)
    


## Podcast Name, Genre & Episode Title 🎙️

There’s something a bit unusual happening with the `Podcast_Name`, `Genre`, and `Episode_Title columns`:
- Combinations of `Podcast_Name` and `Episode_Title` appear multiple times across the dataset.
- What’s odd is that these same combinations are sometimes associated with different `Genres`.

🤔 This raises a few possibilities:
- Could this reflect multiple seasons or re-releases of the same episode under a different category?
- Or is this simply a quirk introduced by the synthetic generation of the dataset?
(Note: this pattern also appears in the original dataset, which is likely also synthetic!)

Interestingly:
- On their own, neither `Genre` nor `Podcast_Name` appear to be strong predictors of `Listening_Time_minutes`.
- But when combined, some distinct patterns emerge, suggesting there might be value in creating interaction features (e.g., `Podcast_Genre_Combo`).


```python
# Count unique genres per podcast name
genre_check = df_train.groupby("Podcast_Name")["Genre"].nunique()

# Filter podcasts that have more than one unique genre
multi_genre_podcasts = genre_check[genre_check > 1]

if multi_genre_podcasts.empty:
    print("All Podcast_Name entries have a consistent Genre.")
else:
    print("Some Podcast_Name entries have multiple Genres assigned:")
    print(multi_genre_podcasts)
```

    Some Podcast_Name entries have multiple Genres assigned:
    Podcast_Name
    Athlete's Arena        10
    Brain Boost             9
    Business Briefs         5
    Business Insights       7
    Comedy Corner           4
    Crime Chronicles       10
    Criminal Minds         10
    Current Affairs         9
    Daily Digest            9
    Detective Diaries       8
    Digital Digest          9
    Educational Nuggets     9
    Fashion Forward         9
    Finance Focus          10
    Fitness First          10
    Funny Folks            10
    Gadget Geek             9
    Game Day               10
    Global News             8
    Health Hour             4
    Healthy Living          9
    Home & Living           9
    Humor Hub              10
    Innovators              9
    Joke Junction          10
    Laugh Line             10
    Learning Lab            9
    Life Lessons            9
    Lifestyle Lounge        8
    Market Masters          9
    Melody Mix              9
    Mind & Body             9
    Money Matters          10
    Music Matters           3
    Mystery Matters        10
    News Roundup            3
    Sound Waves             9
    Sport Spot              9
    Sports Central          5
    Sports Weekly           5
    Study Sessions         10
    Style Guide            10
    Tech Talks             10
    Tech Trends             8
    True Crime Stories      7
    Tune Time              10
    Wellness Wave           9
    World Watch            10
    Name: Genre, dtype: int64
    


```python
fig, axes = plt.subplots(1, 2, figsize=(20, 4), sharey=True)

# 1: Podcast name
ax = axes[0]
sns.boxplot(data=df_train, x="Podcast_Name", y="Listening_Time_minutes", ax=ax, palette="husl")
ax.set_title("Podcast Name")
ax.set_xlabel("")  # Optional: Clean up x-axis label
ax.tick_params(axis='x', rotation=45)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 2: Podcast name
ax = axes[1]
sns.boxplot(data=df_train, x="Genre", y="Listening_Time_minutes", ax=ax,palette="husl")
ax.set_title("Genre")
ax.set_xlabel("")  # Optional: Clean up x-axis label
ax.tick_params(axis='x', rotation=45)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.show()
```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_15_0.png)
    



```python
from math import ceil

# 1. Get unique podcast names and genres
podcast_names = df_train["Podcast_Name"].unique()
n_names = len(podcast_names)

# 2. Determine fixed genre order (sorted by mean listening time across all podcasts)
genre_order = df_train.groupby("Genre")["Listening_Time_minutes"].mean().sort_values().index.tolist()

# 3. Create subplot grid with 2 columns
n_rows = ceil(n_names / 2)
fig, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows), sharey=True)

# Flatten axes array for easy indexing
axes = axes.flatten()

# 4. Plot each podcast
for i, name in enumerate(podcast_names):
    # Subset and ensure all genres are present by reindexing
    subset = df_train[df_train["Podcast_Name"] == name]
    
    # Reindex by full genre order to include missing genres as empty (NaNs)
    dummy = pd.DataFrame({"Genre": genre_order})
    merged = dummy.merge(subset, on="Genre", how="left")  # keeps all genres

    sns.boxplot(data=merged, x="Genre", y="Listening_Time_minutes", ax=axes[i], palette="husl", order=genre_order)
    axes[i].set_title(f"{name}", fontsize=12)
    axes[i].set_xticklabels(genre_order, rotation=45, ha='center')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_16_0.png)
    


# Episode Number from Title 🔢

Looking more closely at the `Episode_Title`, it's possible to extract an episode number.

**Hypothesis**

Intuitively, one might expect some correlation between episode number and listening time:
- Listeners tuning in to, say, Episode 90 are likely fans and may listen through to the end. Meanwhile, Episode 1 might attract new listeners who quickly decide it’s not for them.
- However, this hypothesized trend does not clearly appear in the data.
  
**Adjusted Analysis**

Since `Listening_Time_minutes `is strongly correlated with `Episode_Length_minutes`, I tried normalizing the target by calculating the fraction of the episode that was listened.But even after this adjustment, there was no obvious correlation between episode number and engagement.


```python
# Extract and convert episode number
df_train["Episode_Number"] = df_train["Episode_Title"].str.extract(r"Episode\s+(\d+)", expand=False).astype(int)
# Bin episode numbers into groups of 10
df_train["Episode_Bin"] = pd.cut(df_train["Episode_Number"], bins=range(0, df_train["Episode_Number"].max() + 10, 10), right=False)
# Add bin center
df_train["Bin_Center"] = df_train["Episode_Bin"].apply(lambda x: (x.left + x.right) / 2)

# Group by Episode_Number and calculate the mean values
df_train["Fraction_Listened"] = df_train["Listening_Time_minutes"] / df_train["Episode_Length_minutes"] 
df_train["Fraction_Listened"] = df_train["Fraction_Listened"].replace(np.inf, np.nan)
episode_stats = df_train.groupby("Bin_Center")[["Episode_Length_minutes", "Listening_Time_minutes", "Fraction_Listened"]].mean().reset_index()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

# Line plot for Episode Length
sns.lineplot(data=episode_stats, x="Bin_Center", y="Episode_Length_minutes", marker="o", ax=axes[0], color=sns.color_palette("husl", 1)[0])
axes[0].set_title("Mean Episode Length by Episode Number")
axes[0].set_ylabel("Episode Length (minutes)")
axes[0].set_xlabel("Episode Number")

# Line plot for Listening Time
sns.lineplot(data=episode_stats, x="Bin_Center", y="Listening_Time_minutes", marker="o", ax=axes[1], color=sns.color_palette("husl", 1)[0])
axes[1].set_title("Mean Listening Time by Episode Number")
axes[1].set_ylabel("Listening Time (minutes)")
axes[1].set_xlabel("Episode Number")

# Line plot for Listening Time
sns.lineplot(data=episode_stats, x="Bin_Center", y="Fraction_Listened", marker="o", ax=axes[2], color=sns.color_palette("husl", 1)[0])
axes[2].set_title("Fraction listened by Episode Number")
axes[2].set_ylabel("Fraction Listened")
axes[2].set_xlabel("Episode Number")

plt.tight_layout()
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_18_0.png)
    


## Other categorical features 🏷️

Let’s explore some of the categorical variables in the dataset: `Publication_Day`, `Publication_Time`, `Genre`, `Episode_Sentiment`

**Observations**

There are some correlations between these features and the target listening_time_minutes, but the average effects appear to be quite small. For example:
- With `Episode_Sentiment`, the mean listening time increases only slightly: from ~44 minutes to ~46 minutes going from a negative to a positive sentiment


```python
# List  categorical features to plot
categorical_features = ["Publication_Day", "Publication_Time", "Genre", "Episode_Sentiment"]

# Order oridnal features
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_train["Publication_Day"] = pd.Categorical(df_train["Publication_Day"], categories=day_order, ordered=True)
time_order = ["Morning", "Afternoon", "Evening", "Night"]
df_train["Publication_Time"] = pd.Categorical(df_train["Publication_Time"], categories=time_order, ordered=True)
sentiment_order = ["Negative", "Neutral", "Positive"]
df_train["Episode_Sentiment"] = pd.Categorical(df_train["Episode_Sentiment"], categories=sentiment_order, ordered=True)

# Plot setup
num_features = len(categorical_features)
fig, axes = plt.subplots(2, num_features, figsize=(6 * num_features, 12), sharey='row')

# Plot each boxplot in a subplot
for i, feature in enumerate(categorical_features):
    sns.boxplot(data=df_train, x=feature, y="Listening_Time_minutes", ax=axes[0, i], palette="husl")
    axes[0, i].set_title(f"Listening Time vs {feature}")
    axes[0, i].tick_params(axis='x', rotation=45)

    means = df_train.groupby(feature)["Listening_Time_minutes"].mean().reset_index()    
    sns.lineplot(data=means, x=feature, y="Listening_Time_minutes", marker="o", ax=axes[1, i], color=sns.color_palette("husl", 1)[0])
    axes[1, i].set_title(f"Mean Listening Time by {feature}")
    axes[1, i].tick_params(axis='x', rotation=45)
    axes[1, i].set_ylim(40, 50)

plt.tight_layout()
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_20_0.png)
    


## Publication time & day 🕒

Let’s take a closer look at `Publication_Day` and `Publication_Time`.

At first glance, these features show little standalone correlation with `Listening_Time_minutes`. However, that might be because — like with `Podcast_Name` and `Genre` — their impact is more nuanced and dependent on interactions.

For example:
- One might expect that podcasts released late at night or during work hours could perform differently on weekends vs weekdays.
- Perhaps content published on Friday evenings or Sunday mornings has a different engagement profile compared to midweek or early morning drops.

However, after examining combinations of `Publication_Day` × `Publication_Time`, no clear trend emerges from the data. Listening behavior doesn’t appear to shift significantly depending on these timing combinations.


```python

# Set the day order and time of day order
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
time_order = ["Morning", "Afternoon", "Evening", "Night"]

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for i, time in enumerate(time_order):
    ax = axes[i]
    subset = df_train[df_train["Publication_Time"] == time]

    sns.boxplot(
        data=subset,
        x="Publication_Day",
        y="Listening_Time_minutes",
        order=day_order,
        ax=ax,
        palette="husl"
    )

    ax.set_title(f"{time}")
    ax.set_xlabel("")  # Optional: Clean up x-axis label
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha='center')  # Center tick labels

fig.suptitle("Listening Time by Day of Week and Time of Day", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_22_0.png)
    


## Nobody likes Ads 📉

Let’s be honest — nobody tunes into a podcast hoping for more ads. So it’s no surprise that the `Number_of_Ads` feature shows a negative correlation with `Listening_Time_minutes`.

**Observations**
- As the number of ads increases, average listening time drops.
- There are a few weird entries and strong outliers, which again seem to reflect the synthetic nature of the dataset (e.g., episodes with an unrealistic number of ads or very short durations with high ad counts).
- When plotting the Ad denisty (ads/minute), the negative trend becomes even clearer — higher ad density strongly correlates with lower engagement.


```python
# Create Ad Density, handling NaNs
df_plot = df_train.copy()
df_plot["Ad_Density"] = df_plot["Number_of_Ads"] / df_plot["Episode_Length_minutes"]

# Binning 'Ad_Density' into categories 
bins = [i/20 for i in range(21)]  
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
labels = [f"{center:.2f}" for center in bin_centers]
df_plot["Ad_Density_Category"] = pd.cut(df_plot["Ad_Density"], bins=bins, labels=labels, right=False)

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 1. Boxplot: Listening Time vs Number of Ads
sns.boxplot(
    data=df_plot,
    x="Number_of_Ads",
    y="Listening_Time_minutes",
    ax=axes[0],
    palette="husl"
)
axes[0].set_title("Listening Time by Number of Ads")
axes[0].set_xlabel("Number of Ads")
axes[0].tick_params(axis='x', rotation=45)

# 2. Boxplot: Listening Time vs Ad Density
sns.boxplot(
    data=df_plot.dropna(subset=["Ad_Density_Category"]),
    x="Ad_Density_Category",
    y="Listening_Time_minutes",
    ax=axes[1],
    palette="husl"
)
axes[1].set_title("Listening Time by Ad Density")
axes[1].set_xlabel("Ad Density (ads per minute - binned)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_24_0.png)
    


## Host & Guest popularity 🌟 

Lastly, let’s explore `Host_Popularity_percentage` and `Guest_Popularity_percentage`.

**Initial Insight:**
- On their own, neither feature shows a strong correlation with `Listening_Time_minutes`.
- This suggests that celebrity status alone isn’t a strong predictor of engagement — at least not in a linear way.

**Interestingly:**
- Podcasts where the host is significantly more popular than the guest tend to have slightly longer listening times.
- This might suggest that audiences are more loyal to the host’s brand, sticking around even when the guest is less well-known.



```python
df_plot = df_train.copy()

df_plot["Guest_Popularity_percentage"] = df_plot["Guest_Popularity_percentage"].clip(lower=1, upper=100)
df_plot["Host_Popularity_percentage"] = df_plot["Host_Popularity_percentage"].clip(lower=1, upper=100)

pop_bins = list(range(1, 101, 10)) + [101]  # [1, 11, 21, ..., 91, 101]
bin_labels = [f"{i}-{i+9}" for i in range(1, 100, 10)]  # '1-10', '11-20', ..., '91-100'

df_plot["Guest_Popularity_Binned"] = pd.cut(df_plot["Guest_Popularity_percentage"], bins=pop_bins, labels=bin_labels, right=False)
df_plot["Host_Popularity_Binned"] = pd.cut(df_plot["Host_Popularity_percentage"], bins=pop_bins, labels=bin_labels, right=False)


fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Guest Popularity
sns.boxplot(data=df_plot,
            x="Guest_Popularity_Binned",
            y="Listening_Time_minutes",
            palette="husl",
            ax=axes[0])
axes[0].set_title("Listening Time by Guest Popularity")
axes[0].set_xlabel("Guest Popularity (%)")
axes[0].tick_params(axis='x', rotation=45)
#axes[0].set_ylim(0, 1)


# Host Popularity
sns.boxplot(data=df_plot,
            x="Host_Popularity_Binned",
            y="Listening_Time_minutes",
            palette="husl",
            ax=axes[1])
axes[1].set_title("Listening Time by Host Popularity")
axes[1].set_xlabel("Host Popularity (%)")
axes[1].tick_params(axis='x', rotation=45)
#axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_26_0.png)
    



```python
# Calculate sum and difference
df_plot["Popularity_Sum"] = df_plot["Guest_Popularity_percentage"] + df_plot["Host_Popularity_percentage"]
df_plot["Popularity_Diff"] = df_plot["Host_Popularity_percentage"] - df_plot["Guest_Popularity_percentage"]

# Sum: range will be 2 to 200
sum_bins = list(range(0, 201, 10))
sum_labels = [f"{i}-{i+9}" for i in range(0, 200, 10)]
df_plot["Popularity_Sum_Binned"] = pd.cut(df_plot["Popularity_Sum"], bins=sum_bins, labels=sum_labels, right=False)

# Diff: range -99 to +99, center at 0
diff_bins = list(range(-100, 110, 10))  # allows for -100 to 100
diff_labels = [f"{i:+}-{i+9:+}" for i in range(-100, 100, 10)]
df_plot["Popularity_Diff_Binned"] = pd.cut(df_plot["Popularity_Diff"], bins=diff_bins, labels=diff_labels, right=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Popularity Sum
sns.boxplot(data=df_plot,
            x="Popularity_Sum_Binned",
            y="Listening_Time_minutes",
            palette="husl",
            ax=axes[0])
axes[0].set_title("Listening Time by Popularity Sum (Binned)")
axes[0].set_xlabel("Host + Guest Popularity (%)")
axes[0].tick_params(axis='x', rotation=45)

# Popularity Difference
sns.boxplot(data=df_plot,
            x="Popularity_Diff_Binned",
            y="Listening_Time_minutes",
            palette="husl",
            ax=axes[1])
axes[1].set_title("Listening Time by Popularity Difference (Binned)")
axes[1].set_xlabel("Host - Guest Popularity (%)")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_27_0.png)
    


# Data Preprocessing Pipeline 🛠️

In this section, I define the data preprocessing pipeline using the scikit-learn library. The pipeline is designed to prepare the data for both the XGBoost and PyTorch neural network models, with some steps being model-specific.

🔄 Key Steps
1. Outlier Removal: Filter out extreme and clearly unrealistic values
2. Missing Value Imputation: Imputation is optional and only applied where necessary. XGBoost handles missing values internally, so imputation is mostly relevant for the neural network.
3. Feature Engineering: New features are created based on earlier EDA (e.g., ads_per_minute, popularity_gap, Podcast_Genre_Combo, etc.).
4. Scaling & Encoding: Required for the neural network, which expects numerical inputs and benefits from standardized features. Numerical features are scaled (e.g., StandardScaler). Categorical features are encoded (e.g., OrdinalEncoder or embedding-friendly mappings). XGBoost does not require scaling or encoding — it works directly with raw data and categorical columns (if supported).

🧩 Output Format
- For XGBoost, the processor returns a single DataFrame with all features combined.
- For the neural network, the processor returns two separate DataFrames:
    - One for categorical features (for embeddings)
    - One for numerical features (for direct input after scaling)


```python
from sklearn.base import BaseEstimator, TransformerMixin
```


```python
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_outliers=True):
        self.remove_outliers = remove_outliers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if self.remove_outliers:
            X_copy["Guest_Popularity_percentage"] = X_copy["Guest_Popularity_percentage"].clip(lower=1, upper=100)
            X_copy["Host_Popularity_percentage"] = X_copy["Host_Popularity_percentage"].clip(lower=1, upper=100)
            X_copy["Episode_Length_minutes"] = X_copy["Episode_Length_minutes"].clip(lower=2, upper=120)
            X_copy["Number_of_Ads"] = X_copy["Number_of_Ads"].clip(lower=0, upper=3)
            
        return X_copy
```


```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_missing=True, impute_simple=True):
        self.impute_missing = impute_missing # True or False
        self.impute_simple = impute_simple 
        # Store median values for imputing
        self.num_medians = None # medians of all numerical columns
        self.group_medians_episode = None # medians Epiosode_length_minutes grouped by Genre and Podcast_Title
        self.group_medians_guest = None # medians Guest_Popularity_percentage grouped by Genre 
        # Store numerical and categorical collumn names
        self.num_cols = None  
        self.cat_cols = None 
    
    def fit(self, X, y=None):

        X_copy = X.copy()
        
        # Identify numerical and categorical columns
        self.num_cols = X_copy.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X_copy.select_dtypes(include=["object", "category"]).columns.tolist()

        # Compute and store median of numerical columns for imputation
        self.num_medians = X_copy[self.num_cols].median()

        if not self.impute_simple:
            # Groupwise medians
            self.group_medians_episode = (
                X_copy.groupby(['Podcast_Name', 'Genre'])['Episode_Length_minutes']
                .median()
                .reset_index()
                .rename(columns={'Episode_Length_minutes': 'Median_Episode_Length'})
            )
    
            self.group_medians_guest = (
                X_copy.groupby('Genre')['Guest_Popularity_percentage']
                .median()
                .reset_index()
                .rename(columns={'Guest_Popularity_percentage': 'Median_Guest_Popularity'})
            )

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Impute Episode_Length_minutes and Guest_Popularity_percentage
        if self.impute_missing and not self.impute_simple:
            
            # Merge group medians for episode lenght
            X_copy = X_copy.merge(self.group_medians_episode,how='left',on=['Podcast_Name', 'Genre'])
            X_copy['Episode_Length_minutes'] = X_copy['Episode_Length_minutes'].fillna(X_copy['Median_Episode_Length'])
            X_copy = X_copy.drop(columns='Median_Episode_Length')

            # Merge group medians for guest popularity
            X_copy = X_copy.merge(self.group_medians_guest, how='left',on='Genre')
            X_copy['Guest_Popularity_percentage'] = X_copy['Guest_Popularity_percentage'].fillna(X_copy['Median_Guest_Popularity'])
            X_copy = X_copy.drop(columns='Median_Guest_Popularity')
            
        # Impute other missing values
        if self.impute_missing:
            X_copy[self.num_cols] = X_copy[self.num_cols].fillna(self.num_medians)
            X_copy[self.cat_cols] = X_copy[self.cat_cols].fillna("Missing")
        
        return X_copy
```


```python
from itertools import combinations

class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_cat_combi=True, add_others=True):
        self.add_cat_combi = add_cat_combi # Add all combinations of categorical features
        self.add_others = add_others # add other engineered features
        self.cat_cols = None
        
    def fit(self, X, y=None):
        # Identify categorical cols
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return self


    def transform(self, X):
        X_copy = X.copy()

        if self.add_cat_combi:
            for col1, col2 in combinations(self.cat_cols, 2):
                new_col_name = f"{col1}_{col2}"
                X_copy[new_col_name] = X_copy[col1].astype(str) + "_" + X_copy[col2].astype(str)

        if self.add_others:
            X_copy["Ad_Density"] = X_copy["Number_of_Ads"] / X_copy["Episode_Length_minutes"]
            X_copy["Episode_Number"] = X_copy["Episode_Title"].str.extract(r"Episode\s+(\d+)", expand=False).astype(int)
            X_copy["Popularity_Diff"] = X_copy["Host_Popularity_percentage"] - X_copy["Guest_Popularity_percentage"]
            if not self.add_cat_combi:
                #X_copy["Podcast_Name_Genre"] = X_copy["Podcast_Name"].astype(str) + "_" + X_copy["Genre"].astype(str)
                pass
                
        return X_copy
```


```python
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class GeneralPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=StandardScaler(), impute_missing=True, label_encode=True, split_cat_num=True):
        self.scaler = scaler
        self.impute_missing = impute_missing
        self.label_encode = label_encode
        self.split_cat_num = split_cat_num

        self.num_cols = None
        self.cat_cols = None
        self.num_means = None
        self.cat_encoder = None

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Identify column types
        self.num_cols = X_copy.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X_copy.select_dtypes(include=["object", "category"]).columns.tolist()

        # Compute numerical imputation values
        self.num_means = X_copy[self.num_cols].median()

        # Fit scaler
        if self.scaler:
            self.scaler.fit(X_copy[self.num_cols].fillna(self.num_means))

        # Fit ordinal encoder with safe handling of unknowns
        if self.label_encode:
            self.cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan) 
            self.cat_encoder.fit(X_copy[self.cat_cols].fillna("Missing"))

        return self

    def transform(self, X):
        X_copy = X.copy()

        # Impute missing numerical values
        if self.impute_missing:
            X_copy[self.num_cols] = X_copy[self.num_cols].fillna(self.num_means)
            X_copy[self.cat_cols] = X_copy[self.cat_cols].fillna("Missing")

        # Scale numerical features
        if self.scaler:
            X_copy[self.num_cols] = self.scaler.transform(X_copy[self.num_cols])

        # Encode categorical features
        if self.label_encode:
            X_copy[self.cat_cols] = self.cat_encoder.transform(X_copy[self.cat_cols])
        else:
            X_copy[self.cat_cols] = X_copy[self.cat_cols].astype("category")

        # Return numerical and categorical columns separately or combined
        if self.split_cat_num:
            return X_copy[self.num_cols], X_copy[self.cat_cols]
        else:
            return X_copy

```


```python
class FeatureSelector(BaseEstimator, TransformerMixin):   
    def __init__(self, remove_features=True, features_to_remove=None, features_to_keep=None):
        self.remove_features = remove_features # Remove features_to_remove, if False keep features_to_keep
        self.features_to_remove = features_to_remove
        self.features_to_keep = features_to_keep
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()

        if self.remove_features:
            cols = list(X_copy.columns) 
            cols_to_keep =  [col for col in cols if col not in self.features_to_remove]
            return X_copy[cols_to_keep]
        else:
            return X_copy[ self.features_to_keep]
```


```python
from sklearn.pipeline import Pipeline

# Define the features to remove
cols_to_remove = ["id", "Listening_Time_minutes"]

# Preprocessing for XGBoost model
feature_engineering_pipeline_xgb = Pipeline(steps=[
    ("Cleaning", DataCleaner(remove_outliers=True)),
    ("Imputation", CustomImputer(impute_missing=False, impute_simple=False)),
    ("Feature_engineering", AttributeAdder(add_cat_combi=False, add_others=False)),
])

preprocessor_xgboost = Pipeline(steps=[ # Chain feature engineering, selection and preprocessing
    ("feature_engineering", feature_engineering_pipeline_xgb),
    ("feature_selection", FeatureSelector(remove_features=True, features_to_remove=cols_to_remove)),
    ("preprocessing", GeneralPreprocessor(scaler=None, impute_missing=False, label_encode=False, split_cat_num=False))
])


# Preprocessing for NN model
feature_engineering_pipeline_NN = Pipeline(steps=[
    ("Cleaning", DataCleaner(remove_outliers=True)),
    ("Imputation", CustomImputer(impute_missing=True, impute_simple=True)),
    ("Feature_engineering", AttributeAdder(add_cat_combi=False, add_others=True)),
])

preprocessor_NN = Pipeline(steps=[ # fill missing values for engineered features??
    ("feature_engineering", feature_engineering_pipeline_NN),
    ("feature_selection", FeatureSelector(remove_features=True, features_to_remove=cols_to_remove)),
    ("preprocessing", GeneralPreprocessor(scaler=StandardScaler(), impute_missing=True, label_encode=True, split_cat_num=True))
])
```


```python
# Check preprocessing pipeline
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

X_train_num, X_train_cat = preprocessor_NN.fit_transform(df_train)
X_train_num.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Episode_Length_minutes</th>
      <th>Host_Popularity_percentage</th>
      <th>Guest_Popularity_percentage</th>
      <th>Number_of_Ads</th>
      <th>Ad_Density</th>
      <th>Episode_Number</th>
      <th>Popularity_Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.018947</td>
      <td>0.653654</td>
      <td>0.042273</td>
      <td>-1.213263</td>
      <td>-0.646657</td>
      <td>1.657582</td>
      <td>0.408844</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.786473</td>
      <td>0.310006</td>
      <td>0.918477</td>
      <td>0.586959</td>
      <td>-0.322320</td>
      <td>-0.906009</td>
      <td>-0.482160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.305616</td>
      <td>0.442044</td>
      <td>-1.705044</td>
      <td>-1.213263</td>
      <td>-0.646657</td>
      <td>-1.262063</td>
      <td>1.581031</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.088488</td>
      <td>-0.115401</td>
      <td>1.026191</td>
      <td>0.586959</td>
      <td>-0.068190</td>
      <td>-0.229506</td>
      <td>-0.849997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.486753</td>
      <td>0.883628</td>
      <td>0.242034</td>
      <td>1.487069</td>
      <td>-0.119253</td>
      <td>1.230317</td>
      <td>0.413560</td>
    </tr>
  </tbody>
</table>
</div>



# The Loss Function


```python
# Define loss function
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # Prevent issues when mse=0
        
    def forward(self, y_h, y):
        loss =  torch.sqrt(self.mse(y_h, y) + self.eps)
        return loss

loss_func = RMSELoss()
```

# Training and predictions with XGBoost 🚂

In this section, I’ll walk through the process of training, fitting, and making predictions with an XGBoost model.

🔄 **Cross-Validation Setup**
- To ensure robust model performance, I’m using K-Fold cross-validation. This helps prevent overfitting and gives a more reliable estimate of the model's generalization ability.
- Preprocessing steps (like scaling, encoding, and imputation) are performed within the cross-validation loop to prevent data leakage between training and validation sets.

⚙️ **Model Setup and Training**
- To fine-tune training, I’ll use a learning rate scheduler that progressively reduces the learning rate during training. This often helps the model converge more efficiently and avoid overshooting optimal solutions.

🔮 **Making Predictions**
- After training, predictions are made on the test set (and out-of-fold validation set). These are then used to calculate the final evaluation metric (RMSE).


```python
# Learning rate scheduler
import xgboost as xgb

class CustomLRScheduler(xgb.callback.TrainingCallback):
    def __init__(self, factor=0.5, patience=20, min_lr=1e-5, start_lr=0.01):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.start_lr = start_lr
        self.wait = 0
        self.best_score = float("inf")
        self.current_lr = start_lr

    def before_training(self, model):
        model.set_param("learning_rate", self.current_lr)
        return model

    def after_iteration(self, model, epoch, evals_log):
        score = evals_log["validation_0"]["rmse"][-1]
        if score < self.best_score:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                if new_lr < self.current_lr:
                    print(f"Reducing learning rate to {new_lr:.6f}")
                    self.current_lr = new_lr
                    model.set_param("learning_rate", self.current_lr)
                self.wait = 0
        return False
```


```python
 # Training function for xgboost model
def fit_predict_xgboost_KFold(model, kf, train_df, test_df):
    oof = np.zeros(len(train_df))
    pred = np.zeros(len(test_df))

    # Main loop
    for i, (train_index, val_index) in enumerate(kf.split(train_df)):
    
        print("#"*100)
        print(f"### Fold {i+1}")
        print("#"*100)

         # Split features and targets training set
        X = train_df.drop(columns=["Listening_Time_minutes"]).copy()
        y = train_df["Listening_Time_minutes"].copy()
        
        # Prepare training data
        X_train = X.loc[train_index, :]
        X_train_transformed = preprocessor_xgboost.fit_transform(X_train)
        y_train = y.loc[train_index].copy()
            
        # Prepare validation data
        X_val = X.loc[val_index, :]
        X_val_transformed = preprocessor_xgboost.transform(X_val)
        y_val = y.loc[val_index].copy()

        # Prepare test data
        X_test_transformed = preprocessor_xgboost.transform(test_df)

        # Set callbacks
        model.set_params(callbacks=[
            EarlyStopping(rounds=100, save_best=True),
            CustomLRScheduler(factor=0.5, patience=20, min_lr=1e-5, start_lr=0.1)])
       
        # Fit model
        model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=200)
        
        # Inner oof (out-of-fold predictions)
        oof[val_index] += model.predict(X_val_transformed)
        # Inner test (test predictions)
        pred += model.predict(X_test_transformed)

    # Calculate average predictions
    pred /= kf.get_n_splits() 

    return oof, pred
```


```python
from sklearn.model_selection import KFold
from xgboost.callback import EarlyStopping
import xgboost as xgb

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Model
params = {'max_depth': 14, 
          'learning_rate': 0.1, 
          'n_estimators': 2000, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8,
          'min_child_weight': 40, 
          'enable_categorical': True,
          'tree_method': 'hist',
          'objective':'reg:squarederror',
          'eval_metric':'rmse'}

model_xgb = xgb.XGBRegressor(**params)


oof_xgb, pred_xgb = fit_predict_xgboost_KFold(model_xgb, kf, df_train, df_test)   
```

    ####################################################################################################
    ### Fold 1
    ####################################################################################################
    [0]	validation_0-rmse:25.07732
    Reducing learning rate to 0.050000
    Reducing learning rate to 0.025000
    Reducing learning rate to 0.012500
    Reducing learning rate to 0.006250
    Reducing learning rate to 0.003125
    [194]	validation_0-rmse:12.85124
    ####################################################################################################
    ### Fold 2
    ####################################################################################################
    [0]	validation_0-rmse:25.10913
    Reducing learning rate to 0.050000
    Reducing learning rate to 0.025000
    [200]	validation_0-rmse:12.88416
    Reducing learning rate to 0.012500
    Reducing learning rate to 0.006250
    [400]	validation_0-rmse:12.87789
    [600]	validation_0-rmse:12.87594
    Reducing learning rate to 0.003125
    Reducing learning rate to 0.001563
    Reducing learning rate to 0.000781
    Reducing learning rate to 0.000391
    Reducing learning rate to 0.000195
    [703]	validation_0-rmse:12.87586
    ####################################################################################################
    ### Fold 3
    ####################################################################################################
    [0]	validation_0-rmse:25.08270
    Reducing learning rate to 0.050000
    Reducing learning rate to 0.025000
    Reducing learning rate to 0.012500
    Reducing learning rate to 0.006250
    [200]	validation_0-rmse:12.89198
    [202]	validation_0-rmse:12.89189
    ####################################################################################################
    ### Fold 4
    ####################################################################################################
    [0]	validation_0-rmse:25.14144
    Reducing learning rate to 0.050000
    Reducing learning rate to 0.025000
    Reducing learning rate to 0.012500
    Reducing learning rate to 0.006250
    [187]	validation_0-rmse:12.92164
    ####################################################################################################
    ### Fold 5
    ####################################################################################################
    [0]	validation_0-rmse:25.04508
    Reducing learning rate to 0.050000
    Reducing learning rate to 0.025000
    [200]	validation_0-rmse:12.84854
    Reducing learning rate to 0.012500
    Reducing learning rate to 0.006250
    [400]	validation_0-rmse:12.84574
    Reducing learning rate to 0.003125
    Reducing learning rate to 0.001563
    Reducing learning rate to 0.000781
    Reducing learning rate to 0.000391
    [600]	validation_0-rmse:12.84541
    Reducing learning rate to 0.000195
    Reducing learning rate to 0.000098
    Reducing learning rate to 0.000049
    Reducing learning rate to 0.000024
    [800]	validation_0-rmse:12.84539
    Reducing learning rate to 0.000012
    Reducing learning rate to 0.000010
    [1000]	validation_0-rmse:12.84539
    [1200]	validation_0-rmse:12.84538
    [1400]	validation_0-rmse:12.84538
    [1600]	validation_0-rmse:12.84538
    [1800]	validation_0-rmse:12.84537
    [1999]	validation_0-rmse:12.84537
    

# Training and predictions with PyTorch Neural Network 🤖

In this section, I will walk through the process of training, fitting, and predicting with a PyTorch neural network (NN), taking a similar approach to the XGBoost model, with K-Fold cross-validation and a learning rate scheduler.

⚙️ **Model Setup and Training**
- PyTorch NN Architecture: The setup of the training loop (excluding the cross-validation) is inspired by the 'PyTorch Computer Vision Cookbook', which provides a flexible and scalable approach for training deep learning models.
- Model Architecture: The neural network architecture is based on the work of MuQingyu666 (https://www.kaggle.com/code/muqingyu666/feature-engineering-tabularnn-approach)


```python
# Custom Dataset
class PodcastDataset(Dataset):
    def __init__(self, X_cats, X_nums, y=None): # pass in DataFrames
        self.X_cats = torch.tensor(X_cats.values, dtype=torch.long)  # For embeddings
        self.X_nums = torch.tensor(X_nums.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1) if y is not None else None # None for test set
    
    def __len__(self):
        return len(self.X_cats)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cats[idx], self.X_nums[idx], self.y[idx]
        else: 
            return self.X_cats[idx], self.X_nums[idx]
```


```python
# Custom NN model
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, embedding_sizes, num_numeric, hidden_sizes=[128, 64, 32]):
        super(Net, self).__init__()

        # Embedding layers: list of (num_categories, embedding_dim)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_sizes
        ])

        self.emb_output_dim = sum([emb_dim for _, emb_dim in embedding_sizes]) # Add 1 for unknown values, , padding_idx=-1

        # NN layers after embedding
        self.layers = nn.Sequential(
            nn.Linear(self.emb_output_dim + num_numeric, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[2], 1),
        )

    def forward(self, x_cat, x_num):
        # Apply embedding
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Concatenate numerical and embeddind features
        x = torch.cat(embedded + [x_num], dim=1)
        # Psss trough layers
        x = self.layers(x)
        return x  
```


```python
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
        

def loss_batch(loss_func, yb, yb_h, device, opt=None):
    # Obtain loss
    yb, yb_h = yb.to(device), yb_h.to(device)
    loss = loss_func(yb_h, yb)
    
    if opt:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def loss_epoch(model, loss_func, dataset_loader, device, opt=None):
    loss = 0.
    len_data = len(dataset_loader) # number of batches
    
    for xb_cat, xb_num, yb in dataset_loader: # loop over batches
        xb_cat, xb_num, yb = xb_cat.to(device), xb_num.to(device), yb.to(device)
        yb_h = model(xb_cat, xb_num) # forward pass
        loss_b = loss_batch(loss_func, yb, yb_h, device, opt)
        loss += loss_b
        
    loss /= len_data
    return loss


def train_val(model, params):
    # Unpack parameters
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    device = params["device"]

    # History of loss values
    loss_history={"train": [],"val": []}
    # Deep copy of best model
    best_model_wts = copy.deepcopy(model.state_dict())
    # Initialize best loss to a large value
    best_loss = float('inf')

    # Main loop
    for epoch in range(num_epochs):

        # Get learning rate
        current_lr  = get_lr(opt)
        print(f"epoch:{epoch}, learning rate: {current_lr}")

        # Training
        model.train()
        train_loss = loss_epoch(model, loss_func, train_dl, device, opt)
        # Collect loss value
        loss_history["train"].append(train_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, loss_func, val_dl, device)
        # Collect loss value
        loss_history["val"].append(val_loss)

        # Store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Copied best model weights!")

        # Learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt): # Re-start training with last best model
            print("loading best model weights!")
            model.load_state_dict(best_model_wts)

        print(f"train loss: {train_loss}, val loss: {val_loss}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history


def fit_predict_NN_kFold(kf, params):
    # Unpack parameters
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    train_df = params["train_df"]
    test_df = params["test_df"]

    # Initialize 'oof' and 'pred' to store the predictions
    oof = np.zeros(len(train_df))
    pred = np.zeros(len(test_df))

    # Main loop
    for i, (train_index, val_index) in enumerate(kf.split(train_df)):
    
        print("#"*100)
        print(f"### Fold {i+1}")
        print("#"*100)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Split features and targets training set
        X = train_df.drop(columns=["Listening_Time_minutes"]).copy()
        y = train_df["Listening_Time_minutes"].copy()

        # Split and prepare training and validataion data
        X_train = X.loc[train_index, :]
        X_train_num, X_train_cat = preprocessor_NN.fit_transform(X_train)
        y_train = y.loc[train_index].copy()
        train_dataset = PodcastDataset(X_train_cat, X_train_num, y_train)
        train_dl = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        X_val = X.loc[val_index, :]
        X_val_num, X_val_cat = preprocessor_NN.transform(X_val)
        y_val = y.loc[val_index].copy()
        val_dataset = PodcastDataset(X_val_cat, X_val_num, y_val)
        val_dl = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        # Prepare test data
        X_test_num, X_test_cat = preprocessor_NN.transform(test_df)
        test_dataset = PodcastDataset(X_test_cat, X_test_num)
        test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False)
       
        # Initialize model
        embedding_sizes = [(X_train_cat[col].nunique(), min(50, (X_train_cat[col].nunique() + 1) // 2)) for col in X_train_cat.columns]
        model = Net(embedding_sizes, num_numeric=X_train_num.shape[1])
        model.to(device) 
        # Prepare optimizer and learning rate schedule
        opt = optim.Adam(model.parameters(), lr=1e-3)
        lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        
        # Parameters for training
        params_train = {
            "num_epochs": num_epochs,
            "optimizer": opt,
            "loss_func": loss_func,
            "train_dl": train_dl,
            "val_dl": val_dl,
            "lr_scheduler": lr_scheduler,
            "device": device,
        }
        
        # Training
        model, loss_history = train_val(model, params_train)

        # Evaluation and test predictions
        model.eval()
        with torch.no_grad():
            # Inner oof
            outputs = []
            for xb_cat, xb_num, _ in val_dl:
                xb_cat, xb_num = xb_cat.to(device), xb_num.to(device)
                out = model(xb_cat, xb_num)
                outputs.extend(out.cpu().numpy().flatten())
            oof[val_index] = outputs
            # Inner test
            outputs = []
            for xb_cat, xb_num in test_dl:
                xb_cat, xb_num = xb_cat.to(device), xb_num.to(device)
                out = model(xb_cat, xb_num)
                outputs.extend(out.cpu().numpy().flatten())
            pred += outputs
        
    # Compute average test predictions
    pred /= kf.get_n_splits()

    return oof, pred
```


```python
from sklearn.model_selection import KFold

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

params = {
    "num_epochs": 50,
    "loss_func": loss_func,
    "train_df": df_train,
    "test_df": df_test,
}

oof_NN, pred_NN = fit_predict_NN_kFold(kf, params) 
```

    ####################################################################################################
    ### Fold 1
    ####################################################################################################
    Using device: cpu
    epoch:0, learning rate: 0.001
    Copied best model weights!
    train loss: 39.00342210893338, val loss: 20.32875766883902
    epoch:1, learning rate: 0.001
    Copied best model weights!
    train loss: 15.696988065088162, val loss: 13.348250395586701
    epoch:2, learning rate: 0.001
    Copied best model weights!
    train loss: 14.445291913003238, val loss: 13.264551941229373
    epoch:3, learning rate: 0.001
    Copied best model weights!
    train loss: 14.378959095925602, val loss: 13.256157090063809
    epoch:4, learning rate: 0.001
    Copied best model weights!
    train loss: 14.333396818857551, val loss: 13.201860615996276
    epoch:5, learning rate: 0.001
    train loss: 14.295011471562825, val loss: 13.220600783419448
    epoch:6, learning rate: 0.001
    Copied best model weights!
    train loss: 14.273663465480348, val loss: 13.190207293244447
    epoch:7, learning rate: 0.001
    train loss: 14.266165139325242, val loss: 13.2430227110986
    epoch:8, learning rate: 0.001
    train loss: 14.24742359675645, val loss: 13.238212124831012
    epoch:9, learning rate: 0.001
    train loss: 14.211548484632994, val loss: 13.217136623097115
    epoch:10, learning rate: 0.001
    train loss: 14.208518993325608, val loss: 13.232574352601759
    epoch:11, learning rate: 0.001
    train loss: 14.186364984349586, val loss: 13.228415151842597
    epoch:12, learning rate: 0.001
    loading best model weights!
    train loss: 14.16684897035462, val loss: 13.190758575387553
    epoch:13, learning rate: 0.0005
    Copied best model weights!
    train loss: 14.260387549221312, val loss: 13.179506820886314
    epoch:14, learning rate: 0.0005
    train loss: 14.226509424606688, val loss: 13.182129483644655
    epoch:15, learning rate: 0.0005
    train loss: 14.204406560891318, val loss: 13.195449725300277
    epoch:16, learning rate: 0.0005
    train loss: 14.190010077310504, val loss: 13.180697304861885
    epoch:17, learning rate: 0.0005
    train loss: 14.180674001218517, val loss: 13.180592128208705
    epoch:18, learning rate: 0.0005
    train loss: 14.177080017714777, val loss: 13.185058671600965
    epoch:19, learning rate: 0.0005
    loading best model weights!
    train loss: 14.165554466507947, val loss: 13.186694605820843
    epoch:20, learning rate: 0.00025
    train loss: 14.206797974101512, val loss: 13.187671674352114
    epoch:21, learning rate: 0.00025
    train loss: 14.189723018086404, val loss: 13.186193790565543
    epoch:22, learning rate: 0.00025
    Copied best model weights!
    train loss: 14.183988468638866, val loss: 13.169117038752757
    epoch:23, learning rate: 0.00025
    train loss: 14.189374907431748, val loss: 13.202309653872536
    epoch:24, learning rate: 0.00025
    train loss: 14.186138166095617, val loss: 13.190418742951893
    epoch:25, learning rate: 0.00025
    Copied best model weights!
    train loss: 14.16411772197424, val loss: 13.154578448963814
    epoch:26, learning rate: 0.00025
    train loss: 14.175532451668698, val loss: 13.221769313423001
    epoch:27, learning rate: 0.00025
    train loss: 14.17014030222193, val loss: 13.162615302468645
    epoch:28, learning rate: 0.00025
    train loss: 14.16657546111748, val loss: 13.191249801999046
    epoch:29, learning rate: 0.00025
    train loss: 14.14645996679625, val loss: 13.194228924861571
    epoch:30, learning rate: 0.00025
    train loss: 14.141595290382568, val loss: 13.170920112506062
    epoch:31, learning rate: 0.00025
    loading best model weights!
    train loss: 14.136736199310615, val loss: 13.186158374864227
    epoch:32, learning rate: 0.000125
    train loss: 14.157194562332622, val loss: 13.158261383471846
    epoch:33, learning rate: 0.000125
    train loss: 14.161473023606648, val loss: 13.164339513194804
    epoch:34, learning rate: 0.000125
    train loss: 14.155187090916032, val loss: 13.167883983274706
    epoch:35, learning rate: 0.000125
    train loss: 14.144731612742557, val loss: 13.170516902897633
    epoch:36, learning rate: 0.000125
    train loss: 14.141700406123347, val loss: 13.163386228133222
    epoch:37, learning rate: 0.000125
    loading best model weights!
    train loss: 14.149360559092447, val loss: 13.163899823921879
    epoch:38, learning rate: 6.25e-05
    train loss: 14.156916292861053, val loss: 13.170172503205384
    epoch:39, learning rate: 6.25e-05
    train loss: 14.158344491756816, val loss: 13.173340706598191
    epoch:40, learning rate: 6.25e-05
    train loss: 14.161169175808746, val loss: 13.169194526412861
    epoch:41, learning rate: 6.25e-05
    train loss: 14.152617867896174, val loss: 13.177969342186337
    epoch:42, learning rate: 6.25e-05
    train loss: 14.159697063953804, val loss: 13.1715111700045
    epoch:43, learning rate: 6.25e-05
    loading best model weights!
    train loss: 14.157583139048501, val loss: 13.187158766246977
    epoch:44, learning rate: 3.125e-05
    train loss: 14.172429911512564, val loss: 13.163047316933977
    epoch:45, learning rate: 3.125e-05
    train loss: 14.166427231485933, val loss: 13.187755279800518
    epoch:46, learning rate: 3.125e-05
    train loss: 14.161750629086542, val loss: 13.163104219501522
    epoch:47, learning rate: 3.125e-05
    train loss: 14.141433553077254, val loss: 13.16549809773763
    epoch:48, learning rate: 3.125e-05
    train loss: 14.165482955581092, val loss: 13.18082060132708
    epoch:49, learning rate: 3.125e-05
    loading best model weights!
    train loss: 14.162765475263368, val loss: 13.183371595784921
    ####################################################################################################
    ### Fold 2
    ####################################################################################################
    Using device: cpu
    epoch:0, learning rate: 0.001
    Copied best model weights!
    train loss: 39.523536828597656, val loss: 20.630346869124846
    epoch:1, learning rate: 0.001
    Copied best model weights!
    train loss: 15.467325690663309, val loss: 13.389698735710715
    epoch:2, learning rate: 0.001
    Copied best model weights!
    train loss: 14.451906161910438, val loss: 13.295322995607545
    epoch:3, learning rate: 0.001
    train loss: 14.362692098975588, val loss: 13.336486232524015
    epoch:4, learning rate: 0.001
    train loss: 14.33482993747594, val loss: 13.299669849629305
    epoch:5, learning rate: 0.001
    train loss: 14.279209355038587, val loss: 13.493865687830919
    epoch:6, learning rate: 0.001
    train loss: 14.24653987754327, val loss: 13.475326090442891
    epoch:7, learning rate: 0.001
    train loss: 14.226247559635306, val loss: 13.491583486803535
    epoch:8, learning rate: 0.001
    loading best model weights!
    train loss: 14.207390925176314, val loss: 13.492404645803024
    epoch:9, learning rate: 0.0005
    train loss: 14.354698161623176, val loss: 13.321527558930066
    epoch:10, learning rate: 0.0005
    train loss: 14.322616264681768, val loss: 13.29708837003124
    epoch:11, learning rate: 0.0005
    train loss: 14.287708788601611, val loss: 13.311143330165319
    epoch:12, learning rate: 0.0005
    train loss: 14.253591451221766, val loss: 13.306136533516606
    epoch:13, learning rate: 0.0005
    train loss: 14.241930538476938, val loss: 13.349425075816459
    epoch:14, learning rate: 0.0005
    loading best model weights!
    train loss: 14.233327548251625, val loss: 13.372467560022056
    epoch:15, learning rate: 0.00025
    train loss: 14.342682561776744, val loss: 13.34142099756773
    epoch:16, learning rate: 0.00025
    train loss: 14.325867817263555, val loss: 13.300839184092826
    epoch:17, learning rate: 0.00025
    train loss: 14.309649695308542, val loss: 13.300431647268283
    epoch:18, learning rate: 0.00025
    train loss: 14.279027888392426, val loss: 13.318646437456819
    epoch:19, learning rate: 0.00025
    Copied best model weights!
    train loss: 14.274240715104971, val loss: 13.286193653028839
    epoch:20, learning rate: 0.00025
    train loss: 14.258129417692842, val loss: 13.309250208796287
    epoch:21, learning rate: 0.00025
    train loss: 14.230197146484063, val loss: 13.297138648779214
    epoch:22, learning rate: 0.00025
    train loss: 14.224551677703857, val loss: 13.340967379459718
    epoch:23, learning rate: 0.00025
    train loss: 14.22683121482667, val loss: 13.418731721891026
    epoch:24, learning rate: 0.00025
    train loss: 14.217630348921636, val loss: 13.401510082945531
    epoch:25, learning rate: 0.00025
    loading best model weights!
    train loss: 14.178490682673536, val loss: 13.324106631635809
    epoch:26, learning rate: 0.000125
    train loss: 14.256919312395741, val loss: 13.329040910110992
    epoch:27, learning rate: 0.000125
    train loss: 14.250274340854164, val loss: 13.316655528788663
    epoch:28, learning rate: 0.000125
    train loss: 14.236627669871464, val loss: 13.317481618349245
    epoch:29, learning rate: 0.000125
    train loss: 14.223297073572569, val loss: 13.395501156242526
    epoch:30, learning rate: 0.000125
    train loss: 14.226649751435367, val loss: 13.356847814962167
    epoch:31, learning rate: 0.000125
    loading best model weights!
    train loss: 14.230706434607912, val loss: 13.35010490287729
    epoch:32, learning rate: 6.25e-05
    train loss: 14.236644712324436, val loss: 13.336428752561815
    epoch:33, learning rate: 6.25e-05
    train loss: 14.234713791987188, val loss: 13.343568380187158
    epoch:34, learning rate: 6.25e-05
    train loss: 14.246758470763119, val loss: 13.322049906464661
    epoch:35, learning rate: 6.25e-05
    train loss: 14.228785586438487, val loss: 13.327101454442861
    epoch:36, learning rate: 6.25e-05
    train loss: 14.229371972458354, val loss: 13.320579755873908
    epoch:37, learning rate: 6.25e-05
    loading best model weights!
    train loss: 14.227550114381028, val loss: 13.309298515319824
    epoch:38, learning rate: 3.125e-05
    train loss: 14.241171177743645, val loss: 13.314486081908349
    epoch:39, learning rate: 3.125e-05
    train loss: 14.243803655735055, val loss: 13.303717613220215
    epoch:40, learning rate: 3.125e-05
    train loss: 14.239413005906973, val loss: 13.296308809397171
    epoch:41, learning rate: 3.125e-05
    train loss: 14.235995670227467, val loss: 13.289372755556691
    epoch:42, learning rate: 3.125e-05
    train loss: 14.233759253505147, val loss: 13.293338113901566
    epoch:43, learning rate: 3.125e-05
    loading best model weights!
    train loss: 14.226049997700766, val loss: 13.307238546358485
    epoch:44, learning rate: 1.5625e-05
    train loss: 14.247539352638324, val loss: 13.297246076622788
    epoch:45, learning rate: 1.5625e-05
    train loss: 14.244907553285461, val loss: 13.30034675079138
    epoch:46, learning rate: 1.5625e-05
    train loss: 14.243543890149112, val loss: 13.289817946297783
    epoch:47, learning rate: 1.5625e-05
    train loss: 14.248779329423611, val loss: 13.29009063876405
    epoch:48, learning rate: 1.5625e-05
    train loss: 14.226289485501756, val loss: 13.292118598003777
    epoch:49, learning rate: 1.5625e-05
    loading best model weights!
    train loss: 14.235422682843517, val loss: 13.308570174132885
    ####################################################################################################
    ### Fold 3
    ####################################################################################################
    Using device: cpu
    epoch:0, learning rate: 0.001
    Copied best model weights!
    train loss: 41.69970142312424, val loss: 22.779692461701476
    epoch:1, learning rate: 0.001
    Copied best model weights!
    train loss: 15.917523743756394, val loss: 13.332541426833796
    epoch:2, learning rate: 0.001
    Copied best model weights!
    train loss: 14.445691548109869, val loss: 13.26430842341209
    epoch:3, learning rate: 0.001
    train loss: 14.360328654787239, val loss: 13.344228588804906
    epoch:4, learning rate: 0.001
    train loss: 14.313170415142697, val loss: 13.285763338309566
    epoch:5, learning rate: 0.001
    train loss: 14.270126443674133, val loss: 13.309660476892173
    epoch:6, learning rate: 0.001
    train loss: 14.247543647427607, val loss: 13.264728883496758
    epoch:7, learning rate: 0.001
    Copied best model weights!
    train loss: 14.219464863526536, val loss: 13.256859558780176
    epoch:8, learning rate: 0.001
    train loss: 14.218051518189622, val loss: 13.287833317607438
    epoch:9, learning rate: 0.001
    train loss: 14.195698389828001, val loss: 13.30329483706935
    epoch:10, learning rate: 0.001
    train loss: 14.163348121447775, val loss: 13.29542725102431
    epoch:11, learning rate: 0.001
    Copied best model weights!
    train loss: 14.156357906784214, val loss: 13.255009832836333
    epoch:12, learning rate: 0.001
    Copied best model weights!
    train loss: 14.144611806185987, val loss: 13.247913159480712
    epoch:13, learning rate: 0.001
    Copied best model weights!
    train loss: 14.13964325087876, val loss: 13.236996092763889
    epoch:14, learning rate: 0.001
    train loss: 14.129531876625869, val loss: 13.29217056352265
    epoch:15, learning rate: 0.001
    train loss: 14.11790923453842, val loss: 13.27349919688945
    epoch:16, learning rate: 0.001
    train loss: 14.093480958059786, val loss: 13.280031016083802
    epoch:17, learning rate: 0.001
    train loss: 14.073887006414628, val loss: 13.302758404997741
    epoch:18, learning rate: 0.001
    train loss: 14.065051759062367, val loss: 13.25476480341282
    epoch:19, learning rate: 0.001
    loading best model weights!
    train loss: 14.054196255198924, val loss: 13.280629813265639
    epoch:20, learning rate: 0.0005
    train loss: 14.101359289253939, val loss: 13.312681327871724
    epoch:21, learning rate: 0.0005
    train loss: 14.088474779812548, val loss: 13.249035355185164
    epoch:22, learning rate: 0.0005
    train loss: 14.079934979461566, val loss: 13.275411313893844
    epoch:23, learning rate: 0.0005
    train loss: 14.073925615577567, val loss: 13.2787632844886
    epoch:24, learning rate: 0.0005
    train loss: 14.066813779772344, val loss: 13.241816514203338
    epoch:25, learning rate: 0.0005
    loading best model weights!
    train loss: 14.05829922900672, val loss: 13.292414211091542
    epoch:26, learning rate: 0.00025
    train loss: 14.101887025930775, val loss: 13.26960225007972
    epoch:27, learning rate: 0.00025
    train loss: 14.083728995339456, val loss: 13.277169746606528
    epoch:28, learning rate: 0.00025
    train loss: 14.063624482919739, val loss: 13.237370945158458
    epoch:29, learning rate: 0.00025
    train loss: 14.072208874868451, val loss: 13.2524315840533
    epoch:30, learning rate: 0.00025
    train loss: 14.077363700996894, val loss: 13.273503037537036
    epoch:31, learning rate: 0.00025
    Copied best model weights!
    train loss: 14.069219729192428, val loss: 13.233232537094427
    epoch:32, learning rate: 0.00025
    train loss: 14.044038178570847, val loss: 13.283763794671922
    epoch:33, learning rate: 0.00025
    train loss: 14.05367178965754, val loss: 13.264834682957655
    epoch:34, learning rate: 0.00025
    train loss: 14.046489092270262, val loss: 13.296975213654187
    epoch:35, learning rate: 0.00025
    train loss: 14.04514117940701, val loss: 13.242172487738992
    epoch:36, learning rate: 0.00025
    Copied best model weights!
    train loss: 14.04868711790534, val loss: 13.221315319035329
    epoch:37, learning rate: 0.00025
    train loss: 14.041888733365838, val loss: 13.244696617126465
    epoch:38, learning rate: 0.00025
    train loss: 14.033599409227078, val loss: 13.263329265879936
    epoch:39, learning rate: 0.00025
    train loss: 14.039388363678707, val loss: 13.234826885924047
    epoch:40, learning rate: 0.00025
    train loss: 14.03505156064603, val loss: 13.26448597551203
    epoch:41, learning rate: 0.00025
    train loss: 14.017256121586614, val loss: 13.227403011451774
    epoch:42, learning rate: 0.00025
    loading best model weights!
    train loss: 14.021812380377343, val loss: 13.229317794851704
    epoch:43, learning rate: 0.000125
    train loss: 14.038890853269923, val loss: 13.229226663810055
    epoch:44, learning rate: 0.000125
    train loss: 14.025505742929088, val loss: 13.251247003775875
    epoch:45, learning rate: 0.000125
    train loss: 14.025878727232637, val loss: 13.237454855523142
    epoch:46, learning rate: 0.000125
    train loss: 14.012848414658686, val loss: 13.252429423689032
    epoch:47, learning rate: 0.000125
    train loss: 14.008894106633834, val loss: 13.269914140506666
    epoch:48, learning rate: 0.000125
    loading best model weights!
    train loss: 14.00706380707412, val loss: 13.254297282420048
    epoch:49, learning rate: 6.25e-05
    train loss: 14.032885772783194, val loss: 13.251055075197804
    ####################################################################################################
    ### Fold 4
    ####################################################################################################
    Using device: cpu
    epoch:0, learning rate: 0.001
    Copied best model weights!
    train loss: 39.29193271142224, val loss: 20.54738169300313
    epoch:1, learning rate: 0.001
    Copied best model weights!
    train loss: 15.709917055462, val loss: 13.373477935791016
    epoch:2, learning rate: 0.001
    Copied best model weights!
    train loss: 14.438934814401048, val loss: 13.256565152382365
    epoch:3, learning rate: 0.001
    Copied best model weights!
    train loss: 14.356778284795455, val loss: 13.256196229636263
    epoch:4, learning rate: 0.001
    Copied best model weights!
    train loss: 14.322456804151829, val loss: 13.247519681242858
    epoch:5, learning rate: 0.001
    train loss: 14.282125411180099, val loss: 13.397417237158535
    epoch:6, learning rate: 0.001
    train loss: 14.259971866021791, val loss: 13.266819350573481
    epoch:7, learning rate: 0.001
    Copied best model weights!
    train loss: 14.249589793104361, val loss: 13.236800589529025
    epoch:8, learning rate: 0.001
    train loss: 14.217810100255972, val loss: 13.249100321815128
    epoch:9, learning rate: 0.001
    Copied best model weights!
    train loss: 14.20751428604126, val loss: 13.218558940757699
    epoch:10, learning rate: 0.001
    train loss: 14.198731059507299, val loss: 13.25231935540024
    epoch:11, learning rate: 0.001
    train loss: 14.181924522126494, val loss: 13.222329327849303
    epoch:12, learning rate: 0.001
    train loss: 14.157262629616383, val loss: 13.293893061527589
    epoch:13, learning rate: 0.001
    train loss: 14.136893158479761, val loss: 13.223383105531031
    epoch:14, learning rate: 0.001
    train loss: 14.137512485321878, val loss: 13.240150931741105
    epoch:15, learning rate: 0.001
    loading best model weights!
    train loss: 14.124721463629816, val loss: 13.227385274407004
    epoch:16, learning rate: 0.0005
    Copied best model weights!
    train loss: 14.173974269892982, val loss: 13.21760182802369
    epoch:17, learning rate: 0.0005
    train loss: 14.158778799271827, val loss: 13.281829859934696
    epoch:18, learning rate: 0.0005
    train loss: 14.133845044484318, val loss: 13.224724432238105
    epoch:19, learning rate: 0.0005
    train loss: 14.130221534507672, val loss: 13.270568568690294
    epoch:20, learning rate: 0.0005
    train loss: 14.144917694781828, val loss: 13.2278839747111
    epoch:21, learning rate: 0.0005
    loading best model weights!
    train loss: 14.121294989927637, val loss: 13.27819774264381
    epoch:22, learning rate: 0.00025
    train loss: 14.150322949927007, val loss: 13.235441610115727
    epoch:23, learning rate: 0.00025
    train loss: 14.139917526635701, val loss: 13.232077741298546
    epoch:24, learning rate: 0.00025
    train loss: 14.138090538897206, val loss: 13.255288104621732
    epoch:25, learning rate: 0.00025
    train loss: 14.13784853671598, val loss: 13.271400795501917
    epoch:26, learning rate: 0.00025
    train loss: 14.137023380592007, val loss: 13.245302012177552
    epoch:27, learning rate: 0.00025
    loading best model weights!
    train loss: 14.113518991567982, val loss: 13.258167591224723
    epoch:28, learning rate: 0.000125
    train loss: 14.136311417146754, val loss: 13.232942393036927
    epoch:29, learning rate: 0.000125
    train loss: 14.13099009185114, val loss: 13.224657078178561
    epoch:30, learning rate: 0.000125
    train loss: 14.13439893397048, val loss: 13.225279788581693
    epoch:31, learning rate: 0.000125
    train loss: 14.124140793146127, val loss: 13.23117766737127
    epoch:32, learning rate: 0.000125
    train loss: 14.128963299578775, val loss: 13.229684167978714
    epoch:33, learning rate: 0.000125
    loading best model weights!
    train loss: 14.122265662756389, val loss: 13.239213716416131
    epoch:34, learning rate: 6.25e-05
    train loss: 14.154335052893837, val loss: 13.234779007580816
    epoch:35, learning rate: 6.25e-05
    train loss: 14.131933104869855, val loss: 13.25903287874598
    epoch:36, learning rate: 6.25e-05
    train loss: 14.142788865867328, val loss: 13.236384644800303
    epoch:37, learning rate: 6.25e-05
    train loss: 14.141343803535957, val loss: 13.239215857317658
    epoch:38, learning rate: 6.25e-05
    train loss: 14.12636661692284, val loss: 13.240179237054319
    epoch:39, learning rate: 6.25e-05
    loading best model weights!
    train loss: 14.120525602594578, val loss: 13.251273745582218
    epoch:40, learning rate: 3.125e-05
    train loss: 14.149430291237685, val loss: 13.237361194325143
    epoch:41, learning rate: 3.125e-05
    train loss: 14.136091611491128, val loss: 13.25248977116176
    epoch:42, learning rate: 3.125e-05
    train loss: 14.132765756939051, val loss: 13.24860906276573
    epoch:43, learning rate: 3.125e-05
    train loss: 14.122541989075852, val loss: 13.237129652581247
    epoch:44, learning rate: 3.125e-05
    train loss: 14.128955525342922, val loss: 13.259844909719869
    epoch:45, learning rate: 3.125e-05
    loading best model weights!
    train loss: 14.113498873271226, val loss: 13.233077315246167
    epoch:46, learning rate: 1.5625e-05
    train loss: 14.14178914665769, val loss: 13.223018445125243
    epoch:47, learning rate: 1.5625e-05
    Copied best model weights!
    train loss: 14.145691069319794, val loss: 13.214428343740451
    epoch:48, learning rate: 1.5625e-05
    train loss: 14.14161327186298, val loss: 13.235196950484296
    epoch:49, learning rate: 1.5625e-05
    train loss: 14.13272563589311, val loss: 13.250133430065752
    ####################################################################################################
    ### Fold 5
    ####################################################################################################
    Using device: cpu
    epoch:0, learning rate: 0.001
    Copied best model weights!
    train loss: 40.23122952822532, val loss: 20.65646525791713
    epoch:1, learning rate: 0.001
    Copied best model weights!
    train loss: 15.689298523977754, val loss: 13.299791712339232
    epoch:2, learning rate: 0.001
    Copied best model weights!
    train loss: 14.435086168933648, val loss: 13.245358032434165
    epoch:3, learning rate: 0.001
    train loss: 14.366847334461408, val loss: 13.261874172963253
    epoch:4, learning rate: 0.001
    Copied best model weights!
    train loss: 14.331731130645544, val loss: 13.223855764687467
    epoch:5, learning rate: 0.001
    Copied best model weights!
    train loss: 14.289733816739234, val loss: 13.207946245362159
    epoch:6, learning rate: 0.001
    train loss: 14.261967865680266, val loss: 13.275666353653888
    epoch:7, learning rate: 0.001
    train loss: 14.239944843708859, val loss: 13.295295326077209
    epoch:8, learning rate: 0.001
    train loss: 14.21811328568963, val loss: 13.243355867814044
    epoch:9, learning rate: 0.001
    train loss: 14.210083087556598, val loss: 13.209410284652192
    epoch:10, learning rate: 0.001
    Copied best model weights!
    train loss: 14.190481032934612, val loss: 13.192878989135327
    epoch:11, learning rate: 0.001
    Copied best model weights!
    train loss: 14.17433693465926, val loss: 13.189874019752555
    epoch:12, learning rate: 0.001
    train loss: 14.176219774187627, val loss: 13.241698291026005
    epoch:13, learning rate: 0.001
    train loss: 14.154025474912885, val loss: 13.229197586474776
    epoch:14, learning rate: 0.001
    train loss: 14.125294501464115, val loss: 13.217005872402062
    epoch:15, learning rate: 0.001
    train loss: 14.127833560058281, val loss: 13.247680793814107
    epoch:16, learning rate: 0.001
    train loss: 14.101229272197944, val loss: 13.288001780607262
    epoch:17, learning rate: 0.001
    loading best model weights!
    train loss: 14.099859302768122, val loss: 13.194674465932003
    epoch:18, learning rate: 0.0005
    train loss: 14.151939792437766, val loss: 13.19623057696284
    epoch:19, learning rate: 0.0005
    train loss: 14.132238285533397, val loss: 13.198253119072946
    epoch:20, learning rate: 0.0005
    train loss: 14.11565852897566, val loss: 13.205503282092867
    epoch:21, learning rate: 0.0005
    train loss: 14.11322157374828, val loss: 13.233536512673307
    epoch:22, learning rate: 0.0005
    train loss: 14.1028523477678, val loss: 13.201638111451857
    epoch:23, learning rate: 0.0005
    loading best model weights!
    train loss: 14.096820487910977, val loss: 13.198720795767647
    epoch:24, learning rate: 0.00025
    train loss: 14.132718401964206, val loss: 13.214664569517382
    epoch:25, learning rate: 0.00025
    train loss: 14.122770023020461, val loss: 13.194621053682704
    epoch:26, learning rate: 0.00025
    train loss: 14.119885716422019, val loss: 13.213474682399205
    epoch:27, learning rate: 0.00025
    train loss: 14.102747376868342, val loss: 13.2016797682055
    epoch:28, learning rate: 0.00025
    train loss: 14.098754194409366, val loss: 13.191519283113026
    epoch:29, learning rate: 0.00025
    loading best model weights!
    train loss: 14.1079993947781, val loss: 13.198234454304183
    epoch:30, learning rate: 0.000125
    train loss: 14.122491958605144, val loss: 13.229035416427923
    epoch:31, learning rate: 0.000125
    train loss: 14.125154498494119, val loss: 13.199647053569352
    epoch:32, learning rate: 0.000125
    train loss: 14.110121788832108, val loss: 13.21217577149268
    epoch:33, learning rate: 0.000125
    train loss: 14.115766159096676, val loss: 13.221476931150267
    epoch:34, learning rate: 0.000125
    train loss: 14.104664830217589, val loss: 13.197991144089471
    epoch:35, learning rate: 0.000125
    loading best model weights!
    train loss: 14.1073771092672, val loss: 13.200445823928936
    epoch:36, learning rate: 6.25e-05
    train loss: 14.14370825510383, val loss: 13.210918601678342
    epoch:37, learning rate: 6.25e-05
    train loss: 14.135960412920538, val loss: 13.238166387389306
    epoch:38, learning rate: 6.25e-05
    train loss: 14.118932610078883, val loss: 13.223845572698684
    epoch:39, learning rate: 6.25e-05
    train loss: 14.117924789519847, val loss: 13.198312778862155
    epoch:40, learning rate: 6.25e-05
    train loss: 14.119806375926672, val loss: 13.207334706572448
    epoch:41, learning rate: 6.25e-05
    loading best model weights!
    train loss: 14.114141132237561, val loss: 13.210700859017924
    epoch:42, learning rate: 3.125e-05
    train loss: 14.147080763208175, val loss: 13.20575675834604
    epoch:43, learning rate: 3.125e-05
    train loss: 14.132870091513155, val loss: 13.209220107720823
    epoch:44, learning rate: 3.125e-05
    train loss: 14.131490134541899, val loss: 13.217469578697568
    epoch:45, learning rate: 3.125e-05
    train loss: 14.111130084600871, val loss: 13.197734086691927
    epoch:46, learning rate: 3.125e-05
    train loss: 14.107651635648448, val loss: 13.205086124186613
    epoch:47, learning rate: 3.125e-05
    loading best model weights!
    train loss: 14.12660568159188, val loss: 13.20276132570643
    epoch:48, learning rate: 1.5625e-05
    train loss: 14.139400308042664, val loss: 13.249196688334147
    epoch:49, learning rate: 1.5625e-05
    train loss: 14.131022563973385, val loss: 13.208217692212994
    

# Ensemble & Submit predictions 🤝

In this final step, we combine the predictions from both the XGBoost and PyTorch NN models using ensemble learning and prepare the output for submission.

📊 **Distribution of Predictions**
- Upon inspecting the distribution of the predicted target variable, I noticed some unusual patterns:
- While the distribution of the target in the training set is relatively smooth, the predicted distribution from both the XGBoost and PyTorch NN models shows strange patterns with sharp peaks at certain values.
- Why is this happing?? All suggetsions are welcome!!


```python
# Ensemble OOF predictions validation sets
y_true = torch.tensor(df_train["Listening_Time_minutes"].values, dtype=torch.float32).view(-1, 1)
y_pred_xgb = torch.tensor(oof_xgb, dtype=torch.float32).view(-1, 1)
y_pred_NN = torch.tensor(oof_NN, dtype=torch.float32).view(-1, 1)
y_pred_ens = (y_pred_xgb + y_pred_NN) / 2

loss_oof_xgb = loss_func(y_pred_xgb, y_true)
loss_oof_NN = loss_func(y_pred_NN, y_true)
loss_oof_ens = loss_func(y_pred_ens, y_true)

print(f"OOF loss XGBoost: {loss_oof_xgb.item()}")
print(f"OOF loss NN: {loss_oof_NN.item()}")
print(f"OOF loss Ensemble: {loss_oof_ens.item()}")
```

    OOF loss XGBoost: 12.875570297241211
    OOF loss NN: 13.219209671020508
    OOF loss Ensemble: 12.871585845947266
    


```python
# Ensemble predictions on the test set
pred_ens = (pred_xgb + pred_NN) / 2

result = pd.DataFrame({
    "id": df_test.id,
    "Listening_Time_minutes": pred_ens
})

result.to_csv('submission.csv', index=False)
```


```python
# Visulaize the distribution of the predictions
fig, axes = plt.subplots(2,2, figsize=(8, 6), constrained_layout=True)

axes[0, 0].hist(df_train["Listening_Time_minutes"], bins=range(1, 120))
axes[0, 0].set_title("Training data")
axes[0, 0].set_xlabel("Listening Time (minutes)")
axes[0, 0].set_ylabel("Count")

axes[0, 1].hist(y_pred_ens.numpy(), bins=range(1, 120))
axes[0, 1].set_title("Ensemble predictions")
axes[0, 1].set_xlabel("Listening Time (minutes)")
axes[0, 1].set_ylabel("Count")

axes[1, 0].hist(y_pred_xgb.numpy(), bins=range(1, 120))
axes[1, 0].set_title("XGB predictions")
axes[1, 0].set_xlabel("Listening Time (minutes)")
axes[1, 0].set_ylabel("Count")

axes[1, 1].hist(y_pred_NN.numpy(), bins=range(1, 120))
axes[1, 1].set_title("NN predictions")
axes[1, 1].set_xlabel("Listening Time (minutes)")
axes[1, 1].set_ylabel("Count")

plt.show()
```


    
![png](eda-ensemble-pytorch-xgboost-help-wanted_files/eda-ensemble-pytorch-xgboost-help-wanted_51_0.png)
    

