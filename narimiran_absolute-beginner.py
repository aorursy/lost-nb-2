import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/data.csv')
df.head()

df.shape

random_sample = df.take(np.random.permutation(len(df))[:3])
random_sample.T

not_needed = []

print(df['action_type'].unique())
print(df['combined_shot_type'].unique())

not_needed.extend(['game_event_id', 'game_id'])

sns.set_style('whitegrid')
sns.pairplot(df, vars=['loc_x', 'loc_y', 'lat', 'lon'], hue='shot_made_flag')

not_needed.extend(['lon', 'lat'])

df['time_remaining'] = 60 * df.loc[:, 'minutes_remaining'] + df.loc[:, 'seconds_remaining']

not_needed.extend(['minutes_remaining', 'seconds_remaining'])

df['period'].unique()

df['playoffs'].unique()

df['shot_made_flag'].unique()

df['season'] = df['season'].apply(lambda x: x[:4])
df['season'] = pd.to_numeric(df['season'])

dist = pd.DataFrame({'true_dist': np.sqrt((df['loc_x']/10)**2 + (df['loc_y']/10)**2), 
                     'shot_dist': df['shot_distance']})
dist[:10]

df['shot_distance_'] = dist['true_dist']
not_needed.append('shot_distance')

df['shot_type'].unique()

df['3pt_goal'] = df['shot_type'].str.contains('3PT').astype('int')
not_needed.append('shot_type')

print(df['shot_zone_range'].unique())
print(df['shot_zone_area'].unique())
print(df['shot_zone_basic'].unique())

not_needed.append('shot_zone_range')


area_group = df.groupby('shot_zone_area')
basic_group = df.groupby('shot_zone_basic')

plt.subplots(1, 2, figsize=(15, 7), sharey=True)
colors = list('rgbcmyk')

plt.subplot(121)
plt.ylim(500, -50)
plt.title('shot_zone_area')
for i, (_, area) in enumerate(area_group):
    plt.scatter(area['loc_x'], area['loc_y'], alpha=0.1, color=colors[i])
    
plt.subplot(122)
plt.ylim(500, -50)
plt.title('shot_zone_basic')
for i, (_, basic) in enumerate(basic_group):
    plt.scatter(basic['loc_x'], basic['loc_y'], alpha=0.1, color=colors[i])


print(df['team_id'].unique())
print(df['team_name'].unique())


not_needed.extend(['team_id', 'team_name'])

df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.dayofweek

not_needed.append('game_date')


df['home_game'] = df['matchup'].str.contains('vs.').astype(int)
not_needed.append('matchup')

df.set_index('shot_id', inplace=True)

df = df.drop(not_needed, axis=1)

df.shape

pd.set_option('display.max_columns', None)
random_sample = df.take(np.random.permutation(len(df))[:10])
random_sample.head(10)

submission_data = df[df['shot_made_flag'].isnull()]
submission_data = submission_data.drop('shot_made_flag', 1)
submission_data.shape

data = df[df['shot_made_flag'].notnull()]
data.shape

sns.countplot(x='shot_made_flag', data=data)

data['shot_made_flag'].value_counts() / data['shot_made_flag'].shape

data['time_remaining'].plot(kind='hist', bins=24, xlim=(720, 0), figsize=(12,6),
                            title='Attempts made over time\n(seconds to the end of period)')

time_bins = np.arange(0, 721, 30)
attempts_in_time = pd.cut(data['time_remaining'], time_bins, right=False)
grouped = data.groupby(attempts_in_time)
prec = grouped['shot_made_flag'].mean()

prec[::-1].plot(kind='bar', figsize=(12, 6), ylim=(0.2, 0.5), 
                title='Shot accuracy over time\n(seconds to the end of period)')

last_30 = data[data['time_remaining'] < 30]
last_30['shot_made_flag'].value_counts() / last_30['shot_made_flag'].shape

last_2min = data[data['time_remaining'] <= 120]

last_2min['time_remaining'].plot(kind='hist', bins=30, xlim=(120, 0), figsize=(12,6),
                            title='Attempts made over time\n(seconds to the end of period)')

last_30['time_remaining'].plot(kind='hist', bins=10, xlim=(30, 0), figsize=(12,6),
                            title='Attempts made over time\n(seconds to the end of period)')

last_5sec_misses = data[(data['time_remaining'] <= 5) & (data['shot_made_flag'] == 0)]
last_5sec_scores = data[(data['time_remaining'] <= 5) & (data['shot_made_flag'] == 1)]


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,7))
ax1.set_ylim(800, -50)

sns.regplot(x='loc_x', y='loc_y', data=last_5sec_misses, fit_reg=False, ax=ax1, color='r')
sns.regplot(x='loc_x', y='loc_y', data=last_5sec_scores, fit_reg=False, ax=ax2, color='g')

last_5sec_close = data[(data['time_remaining'] <= 5) & (data['shot_distance_'] <= 20)]

last_5sec_close['shot_made_flag'].value_counts() / last_5sec_close['shot_made_flag'].shape

For comparison, accuracy from close distance when there are more than 5 seconds to go:

close_shots = data[(data['time_remaining'] > 5) & (data['shot_distance_'] <= 20)]

close_shots['shot_made_flag'].value_counts() / close_shots['shot_made_flag'].shape

plt.figure(figsize=(12,6))
sns.countplot(x="period", hue="shot_made_flag", data=data)

period_acc = data['shot_made_flag'].groupby(data['period']).mean()
period_acc.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data)

shot_type_acc = data['shot_made_flag'].groupby(data['combined_shot_type']).mean()
shot_type_acc.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,18))
sns.countplot(y="action_type", hue="shot_made_flag", data=data)

action_type = data['shot_made_flag'].groupby(data['action_type']).mean()
action_type.sort_values()

action_type.sort_values().plot(kind='barh', figsize=(12, 18))

plt.figure(figsize=(12,6))
sns.countplot(x="season", hue="shot_made_flag", data=data)

season_acc = data['shot_made_flag'].groupby(data['season']).mean()
season_acc.plot(figsize=(12, 6), title='Accuracy over seasons')

plt.figure(figsize=(12,6))
sns.countplot(x="game_month", hue="shot_made_flag", data=data)

game_month = data['shot_made_flag'].groupby(data['game_month']).mean()
game_month.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="game_day", hue="shot_made_flag", data=data)

game_day = data['shot_made_flag'].groupby(data['game_day']).mean()
game_day.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="playoffs", hue="shot_made_flag", data=data)

playoffs = data['shot_made_flag'].groupby(data['playoffs']).mean()
playoffs.plot(kind='barh', figsize=(12, 2), xlim=(0, 0.50))

distance_bins = np.append(np.arange(0, 31, 3), 300) 
distance_cat = pd.cut(data['shot_distance_'], distance_bins, right=False)

dist_data = data.loc[:, ['shot_distance_', 'shot_made_flag']]
dist_data['distance_cat'] = distance_cat

distance_cat.value_counts(sort=False)

plt.figure(figsize=(12,6))
sns.countplot(x="distance_cat", hue="shot_made_flag", data=dist_data)

dist_prec = dist_data['shot_made_flag'].groupby(dist_data['distance_cat']).mean()
dist_prec.plot(kind='bar', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data)

shot_area = data['shot_made_flag'].groupby(data['shot_zone_area']).mean()
shot_area.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data)

shot_basic = data['shot_made_flag'].groupby(data['shot_zone_basic']).mean()
shot_basic.plot(kind='barh', figsize=(12, 6))

plt.figure(figsize=(12,6))
sns.countplot(x="home_game", hue="shot_made_flag", data=data)

shot_basic = data['shot_made_flag'].groupby(data['home_game']).mean()
shot_basic.plot(kind='barh', figsize=(12, 2))

plt.figure(figsize=(12,16))
sns.countplot(y="opponent", hue="shot_made_flag", data=data)

opponent = data['shot_made_flag'].groupby(data['opponent']).mean()
opponent.sort_values().plot(kind='barh', figsize=(12,10))


