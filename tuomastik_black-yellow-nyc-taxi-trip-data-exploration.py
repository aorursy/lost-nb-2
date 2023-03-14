#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '\n<!DOCTYPE HTML>\n<html>\n  <head>\n    <meta charset="utf-8" />\n    <style>\n\nbody {\n  padding-bottom: 40px;\n}\n\n#content-container {\n  position: relative;\n  width: 820px;\n  height: 200px;\n}\n\n.car {\n  position: absolute;\n  width: 58px;\n  top: 0px;\n  /* Flip the car horizontally */\n  -moz-transform: scaleX(-1);\n  -webkit-transform: scaleX(-1);\n  -o-transform: scaleX(-1);\n  transform: scaleX(-1);\n  -ms-filter: fliph; /*IE*/\n  filter: fliph; /*IE*/\n  /* Animation */\n  -webkit-animation: car-animation 10s ease infinite; /* Safari 4+ */\n  -moz-animation:    car-animation 10s ease infinite; /* Fx 5+ */\n  -o-animation:      car-animation 10s ease infinite; /* Opera 12+ */\n  animation:         car-animation 10s ease infinite; /* IE 10+, Fx 29+ */\n}\n@-webkit-keyframes car-animation {\n  0%   { opacity: 0; left:   0px; }\n  40%  { opacity: 1; left: 256px; }\n  75%  { opacity: 1; left: 555px; }\n  100% { opacity: 0; left: 777px; }\n}\n@-moz-keyframes car-animation {\n  0%   { opacity: 0; left:   0px; }\n  40%  { opacity: 1; left: 256px; }\n  75%  { opacity: 1; left: 555px; }\n  100% { opacity: 0; left: 777px; }\n}\n@-o-keyframes car-animation {\n  0%   { opacity: 0; left:   0px; }\n  40%  { opacity: 1; left: 256px; }\n  75%  { opacity: 1; left: 555px; }\n  100% { opacity: 0; left: 777px; }\n}\n@keyframes car-animation {\n  0%   { opacity: 0; left:   0px; }\n  40%  { opacity: 1; left: 256px; }\n  75%  { opacity: 1; left: 555px; }\n  100% { opacity: 0; left: 777px; }\n}\n\n\n.passenger-start {\n  position: absolute;\n  width: 30px;\n  top: 11px;\n  -webkit-animation: passenger-start-animation 10s ease infinite; /* Safari 4+ */\n  -moz-animation:    passenger-start-animation 10s ease infinite; /* Fx 5+ */\n  -o-animation:      passenger-start-animation 10s ease infinite; /* Opera 12+ */\n  animation:         passenger-start-animation 10s ease infinite; /* IE 10+, Fx 29+ */\n}\n@-webkit-keyframes passenger-start-animation {\n  0%   { opacity: 0; left: 265px; }\n  20%  { opacity: 1; left: 265px; }\n  40%  { opacity: 1; left: 265px; }\n  45%  { opacity: 0; left: 265px; }\n  100% { opacity: 0; left: 265px; }\n}\n@-moz-keyframes passenger-start-animation {\n  0%   { opacity: 0; left: 265px; }\n  20%  { opacity: 1; left: 265px; }\n  40%  { opacity: 1; left: 265px; }\n  45%  { opacity: 0; left: 265px; }\n  100% { opacity: 0; left: 265px; }\n}\n@-o-keyframes passenger-start-animation {\n  0%   { opacity: 0; left: 265px; }\n  20%  { opacity: 1; left: 265px; }\n  40%  { opacity: 1; left: 265px; }\n  45%  { opacity: 0; left: 265px; }\n  100% { opacity: 0; left: 265px; }\n}\n@keyframes passenger-start-animation {\n  0%   { opacity: 0; left: 265px; }\n  20%  { opacity: 1; left: 265px; }\n  40%  { opacity: 1; left: 265px; }\n  45%  { opacity: 0; left: 265px; }\n  100% { opacity: 0; left: 265px; }\n}\n\n\n.passenger-stop {\n  position: absolute;\n  width: 30px;\n  top: 11px;\n  -webkit-animation: passenger-stop-animation 10s ease infinite; /* Safari 4+ */\n  -moz-animation:    passenger-stop-animation 10s ease infinite; /* Fx 5+ */\n  -o-animation:      passenger-stop-animation 10s ease infinite; /* Opera 12+ */\n  animation:         passenger-stop-animation 10s ease infinite; /* IE 10+, Fx 29+ */\n}\n@-webkit-keyframes passenger-stop-animation {\n  0%   { opacity: 0; left: 564px; }\n  68%  { opacity: 0; left: 564px; }\n  75%  { opacity: 1; left: 564px; }\n  80%  { opacity: 1; left: 564px; }\n  100% { opacity: 0; left: 564px; }\n}\n@-moz-keyframes passenger-stop-animation {\n  0%   { opacity: 0; left: 564px; }\n  68%  { opacity: 0; left: 564px; }\n  75%  { opacity: 1; left: 564px; }\n  80%  { opacity: 1; left: 564px; }\n  100% { opacity: 0; left: 564px; }\n}\n@-o-keyframes passenger-stop-animation {\n  0%   { opacity: 0; left: 564px; }\n  68%  { opacity: 0; left: 564px; }\n  75%  { opacity: 1; left: 564px; }\n  80%  { opacity: 1; left: 564px; }\n  100% { opacity: 0; left: 564px; }\n}\n@keyframes passenger-stop-animation {\n  0%   { opacity: 0; left: 564px; }\n  68%  { opacity: 0; left: 564px; }\n  75%  { opacity: 1; left: 564px; }\n  80%  { opacity: 1; left: 564px; }\n  100% { opacity: 0; left: 564px; }\n}\n\n\n.header {\n  position: absolute;\n  left: 0px;\n  top: 40px;\n  text-align: center;\n  vertical-align: middle;\n  background: #ffb700;\n  width: 820px;\n  height: 163px;\n  border: 10px solid #111111;\n  -webkit-border-radius: 25px 25px 25px 25px;\n  border-radius: 25px 25px 25px 25px; \n}\n\n.competition-name {\n  margin: 0;\n  font-size: 50px;\n  color: #111111;\n  display: inline-block;\n}\n\n.taxi-text {\n  background: #111111;\n  color: #ffb700;\n  border: 10px solid #111111;\n  -moz-border-radius: 0px;\n  -webkit-border-radius: 25px 25px 0px 0px;\n  border-radius: 25px 25px 0px 0px; \n  margin-left: 20px;\n  margin-right: 20px;\n  padding: 7px 15px 0 15px;\n}\n\n\n    </style>\n  </head>\n  <body>\n\n    <div id="content-container">\n      <div>\n        <div>\n          <!-- Icon made by Freepik from www.flaticon.com -->\n          <!-- https://www.flaticon.com/free-icon/taxi_89131 -->\n          <img class="car" src="https://image.flaticon.com/icons/svg/89/89131.svg"/>\n        </div>\n        <div>\n          <!-- Icon made by Freepik from www.flaticon.com -->\n          <!-- https://www.flaticon.com/free-icon/call-taxi_10931 -->\n          <img class="passenger-start" src="https://image.flaticon.com/icons/svg/10/10931.svg"/>\n        </div>\n        <div>\n          <!-- Icon made by Freepik from www.flaticon.com -->\n          <!-- https://www.flaticon.com/free-icon/businessman-with-suitcase_49205 -->\n          <img class="passenger-stop" src="https://image.flaticon.com/icons/svg/49/49205.svg"/>\n        </div>\n      </div>\n\n      <div class="header">\n        <h1 class="competition-name"           style="font-size: 40px; margin-top: 30px;">New York City</h1>\n        <h1 class="competition-name taxi-text" style="font-size: 40px; margin-top: 30px;\n                                                      margin-left: 15px; margin-right: 15px;\n                                                      color: #ffb700;">TAXI</h1>\n        <h1 class="competition-name"           style="font-size: 40px; margin-top: 30px;">Trip Duration</h1>\n      </div>\n    </div>\n\n  </body>\n</html>')


# In[2]:


# Set things up

# Make Matplotlib to work with Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore ugly DeprecationWarnings
from bokeh import BokehDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)

import datetime
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from bokeh.io import output_notebook
from bokeh.charts import show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, DataRange1d, HoverTool, PanTool, WheelZoomTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models.widgets import DateFormatter, StringFormatter
from bokeh.plotting import figure
import folium
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from geopy.distance import great_circle

plt.style.use('ggplot')  # Change Matplotlib style to something nicer
plt.rc('font', size=12)  # Set default Matplotlib font size
output_notebook(hide_banner=True)  # Make Bokeh to work with Jupyter Notebook

# Define colors
COLOR_DARK = "#212121"
COLOR_YELLOW = "#ffb700"
# Define custom colormap
plt.register_cmap(cmap=colors.LinearSegmentedColormap.from_list(
    'TaxiYellow', ["#ffffff", COLOR_YELLOW]))


# In[3]:


train = pd.read_csv(filepath_or_buffer='../input/train.csv', index_col='id',
                    parse_dates=['pickup_datetime', 'dropoff_datetime'],
                    infer_datetime_format=True)

test = pd.read_csv(filepath_or_buffer='../input/test.csv', index_col='id',
                   parse_dates=['pickup_datetime'],
                   infer_datetime_format=True)


# In[4]:


n_train_rows, n_train_cols = train.shape
n_test_rows, n_test_cols = test.shape
print('- Training data has {:9,} rows and {:2,} columns.'.format(*train.shape))
print('- Testing data has  {:9,} rows and {:2,} columns.'.format(*test.shape))
print('- There are {:.1f} times more (#{:,}) training data examples than '
      'testing data examples.'.format(n_train_rows / n_test_rows,
                                        n_train_rows - n_test_rows))
print("- There are %i missing values in the training data." % train.isnull().sum().sum())
print("- There are %i missing values in the testing data." % test.isnull().sum().sum())


# In[5]:


print("The first 10 rows of the training data shown in an interactive Bokeh table:")
show(DataTable(
    source=ColumnDataSource(train.head(10)), editable=True,
    columns=[TableColumn(
        field=c, title=c,
        formatter=DateFormatter() if 'datetime' in c else StringFormatter())
        for c in train.columns],
    width=820, height=300))

print("The first 10 rows of the testing data shown in an interactive Bokeh table:")
show(DataTable(
    source=ColumnDataSource(test.head(10)),
    columns=[TableColumn(
        field=c, title=c,
        formatter=DateFormatter() if 'datetime' in c else StringFormatter())
        for c in test.columns],
    width=820, height=300))

# -----------------------------------------------------------------------------

# Let's visualize the number of examples in each file
ax = pd.DataFrame({'Train': [n_train_rows], 'Test': [n_test_rows]}).plot.barh(
    figsize=(14.1, 2), legend='reverse', rot=90, color=[COLOR_YELLOW, COLOR_DARK])
ax.set(xlabel='Number of examples', ylabel='Dataset')
ax.set_title('The number of examples in each dataset', fontsize=14)
ax.get_yaxis().set_ticks([])
# For readability, add commas to separate thousand in the x-axis labels
ax.set_xticklabels([format(label, ',.0f') for label in ax.get_xticks()]);


# In[6]:


for col_name_pretty, col_name_official, figsize in zip(
        ['passenger count', 'vendor ID', 'store and forward flag'],
        ['passenger_count', 'vendor_id', 'store_and_fwd_flag'],
        [(14.1, 6), (14.1, 2), (14.1, 2)]):
    ax = pd.DataFrame({
        col_name_pretty.capitalize() + ' (train)':
            train[col_name_official].value_counts() /
            train[col_name_official].value_counts().sum() * 100,
        col_name_pretty.capitalize() + ' (test)':
            test[col_name_official].value_counts() /
            test[col_name_official].value_counts().sum() * 100}).plot.barh(
        figsize=figsize, legend='reverse', rot=90, stacked=False,
        color=[COLOR_YELLOW, COLOR_DARK])
    ax.set(xlabel='Percentage of all values', ylabel='Value')
    ax.set_title('Value proportions of %s in training and testing data' % col_name_pretty,
                 fontsize=14)


# In[7]:


for var_name_pretty, var_name_official, datasets, dataset_names, cmaps in zip(
        ["pickup date", "drop-off date"],
        ["pickup_datetime", "dropoff_datetime"],
        # "dropoff_datetime" is not included in the testing data
        [[train, test], [train]], [['train', 'test'], ['train']],
        [[cm.get_cmap('binary'), cm.get_cmap('TaxiYellow')], [cm.get_cmap('binary')]]):

    fig = figure(plot_width=820, plot_height=400, x_axis_type="datetime",
                 tools=('wheel_zoom, pan, save, tap, reset',
                        HoverTool(tooltips=[(var_name_pretty.capitalize(), '@time'),
                                            ('Count', '@Count')])),
                 x_axis_location='below', toolbar_location="above", logo=None,
                 x_axis_label=var_name_pretty.capitalize(), y_axis_label='Count',
                 y_range=(0, 10000), x_range=DataRange1d(range_padding=0.0),
                 title='Counts of %ss in %s data' % (
                     var_name_pretty, ' and '.join(dataset_names)))

    for dataset, dataset_name, cmap in zip(datasets, dataset_names, cmaps):
        # Group date and time counts by years, months and days
        grouped = pd.DataFrame(dataset[var_name_official].groupby(
            [dataset[var_name_official].dt.year,
             dataset[var_name_official].dt.month,
             dataset[var_name_official].dt.day]).count()).rename(
            columns={var_name_official: 'Count'})
        # Convert MultiIndex to datetime
        grouped['date'] = [datetime.datetime(*i) for i in grouped.index]
        # Determine left, right and bottom coordinates of each bar in the plot
        grouped['left'] = grouped['date'] - datetime.timedelta(days=0.5)
        grouped['right'] = grouped['date'] + datetime.timedelta(days=0.5)
        grouped['bottom'] = [0] * grouped.shape[0]
        # Set date as the new index
        grouped.index = grouped['date']
        # Create additional 'time' variable, which is shown in hover tooltip
        grouped['time'] = grouped['date'].map(lambda x: x.strftime('%d %B %Y (%A)'))
        grouped = grouped.sort_index()  # Sort just in case
        # Create colors for bars based on bar height
        norm = colors.Normalize(vmax=grouped['Count'].max(),
                                vmin=grouped['Count'].min() - 2 * grouped['Count'].min())
        bar_colors = [colors.rgb2hex(cmap(norm(c))) for c in grouped['Count']]
        # Draw one bar for each date
        fig.quad(top='Count', bottom='bottom', left='left', right='right',
                 source=ColumnDataSource(grouped), color=bar_colors,
                 legend=dataset_name.capitalize())
    fig.legend.location = 'bottom_right'
    show(fig)


# In[8]:


figs = []
for var_name_pretty, var_name_official, datasets, dataset_names, cmaps in zip(
        ["pickup month", "drop-off month"],
        ["pickup_datetime", "dropoff_datetime"],
        # "dropoff_datetime" is not included in the testing data
        [[train, test], [train]], [['train', 'test'], ['train']],
        [[cm.get_cmap('binary'), cm.get_cmap('TaxiYellow')], [cm.get_cmap('binary')]]):

    fig = figure(plot_width=410, plot_height=400, x_axis_type="datetime",
                 tools=('wheel_zoom, pan, save, tap, reset',
                        HoverTool(tooltips=[(var_name_pretty.capitalize(), '@time'),
                                            ('Count', '@Count')])),
                 x_axis_location='below', toolbar_location="right", logo=None,
                 x_axis_label=var_name_pretty.capitalize(), y_axis_label='Count',
                 y_range=(0, 270000), x_range=DataRange1d(range_padding=0.0),
                 title='Counts of %ss in %s data' % (
                     var_name_pretty, ' and '.join(dataset_names)))

    for dataset, dataset_name, cmap in zip(datasets, dataset_names, cmaps):
        # Group date and time counts by months
        grouped = pd.DataFrame(dataset[var_name_official].groupby(
            dataset[var_name_official].dt.month).count()).rename(
            columns={var_name_official: 'Count'})
        # Convert MultiIndex to datetime
        grouped['date'] = [datetime.datetime(year=2016, month=i, day=1) for i in grouped.index]
        # Determine left, right and bottom coordinates of each bar in the plot
        grouped['left'] = grouped['date']
        grouped['right'] = [d + datetime.timedelta(calendar.monthrange(2016, d.month)[1])
                            for d in grouped['date']]
        grouped['bottom'] = [0] * grouped.shape[0]
        # Set date as the new index
        grouped.index = grouped['date']
        # Create additional 'time' variable, which is shown in hover tooltip
        grouped['time'] = grouped['date'].map(lambda x: x.strftime('%B %Y'))
        grouped = grouped.sort_index()  # Sort just in case
        # Create colors for bars based on bar height
        norm = colors.Normalize(vmax=grouped['Count'].max(),
                                vmin=grouped['Count'].min() - 1 * grouped['Count'].min())
        bar_colors = [colors.rgb2hex(cmap(norm(c))) for c in grouped['Count']]
        # Draw one bar for each month
        fig.quad(top='Count', bottom='bottom', left='left', right='right',
                 source=ColumnDataSource(grouped), color=bar_colors,
                 legend=dataset_name.capitalize())
    fig.legend.location = 'bottom_left'
    figs.append(fig)
show(row(figs))


# In[9]:


for var_name_pretty, var_name_official, datasets, dataset_names, cmaps in zip(
        ["pickup hour", "drop-off hour"],
        ["pickup_datetime", "dropoff_datetime"],
        # "dropoff_datetime" is not included in the testing data
        [[train, test], [train]], [['train', 'test'], ['train']],
        [[cm.get_cmap('binary'), cm.get_cmap('TaxiYellow')], [cm.get_cmap('binary')]]):

    fig = figure(plot_width=820, plot_height=400,
                 tools=('wheel_zoom, pan, save, tap, reset',
                        HoverTool(tooltips=[(var_name_pretty.capitalize(), '$index'),
                                            ('Count', '@Count')])),
                 x_axis_location='below', toolbar_location="above", logo=None,
                 x_axis_label=var_name_pretty.capitalize(), y_axis_label='Count',
                 x_range=(-0.5, 23.5), y_range=(0, 99999),
                 title='Counts of %ss in %s data' % (
                     var_name_pretty, ' and '.join(dataset_names)))

    for dataset, dataset_name, cmap in zip(datasets, dataset_names, cmaps):
        # Group date and time counts by hours
        grouped = pd.DataFrame(dataset[var_name_official].groupby(
            [dataset[var_name_official].dt.hour]).count()).rename(
            columns={var_name_official: 'Count'})
        # Determine left, right and bottom coordinates of each bar in the plot
        grouped['left'] = grouped.index - 0.5
        grouped['right'] = grouped.index + 0.5
        grouped['bottom'] = [0] * grouped.shape[0]
        # Create colors for bars based on bar height
        norm = colors.Normalize(vmax=grouped['Count'].max(),
                                vmin=grouped['Count'].min() - 7 * grouped['Count'].min())
        bar_colors = [colors.rgb2hex(cmap(norm(c))) for c in grouped['Count']]
        # Draw one bar for each hour
        fig.quad(top='Count', bottom='bottom', left='left', right='right',
                 source=ColumnDataSource(grouped), color=bar_colors,
                 legend=dataset_name.capitalize())
    fig.legend.location = 'top_left'
    show(fig)


# In[10]:


trip_dur_mins = train['trip_duration'] / 60
figs = []
for var, var_name, y_range in zip([trip_dur_mins, np.log10(trip_dur_mins)],
                                  ["Trip duration (minutes)", "Log10(Trip duration (minutes))"],
                                  [(0, 15e5), (0, 2.5e5)]):
    # Calculate histogram
    counts, bin_edges = np.histogram(var, bins=50)
    # Generate colors for the histogram bars
    colormap = cm.get_cmap('hot')
    norm = colors.Normalize(vmax=np.max(np.array(counts)) + .5 * np.max(np.array(counts)),
                            vmin=np.min(np.array(counts)))
    hist_colors = [colors.rgb2hex(colormap(norm(n_vals))) for n_vals in counts]
    source = ColumnDataSource(data=dict(top=counts, bottom=[0] * counts.size,
                                        left=bin_edges[:-1], right=bin_edges[1:]))
    fig = figure(width=410, height=300,
                 tools=('wheel_zoom', 'pan', 'save, tap, reset', HoverTool(
                     tooltips=[(var_name.replace(" (minutes)", ""),
                                "@left-@right"), ("Count", "@top")])),
                 x_axis_location='below', toolbar_location="right", logo=None,
                 x_axis_label=var_name, y_axis_label='Count', y_range=y_range,
                 title="Histogram of %s" % var_name.lower())
    fig.quad(source=source, top='top', right='right', left='left', bottom='bottom',
             color=hist_colors)
    figs.append(fig)
show(row(figs))

# Plot the beginning of a boxplot of the original, non-normalized data
# to realize the skewness of the distribution
ax = plt.figure(figsize=(14.1, 3)).add_subplot(111)
boxplot = ax.boxplot(trip_dur_mins, vert=False, patch_artist=True, showfliers=False)
ax.set(xlabel='Minutes', ylabel='Trip duration', xlim=(-5, 40))
ax.get_yaxis().set_ticks([])
for item in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(boxplot[item], color=COLOR_DARK, linewidth=1.5)
plt.setp(boxplot["medians"], color=COLOR_YELLOW, linewidth=3)
plt.title("Box plot of trip duration (minutes) without large outliers.", size=14)
plt.show();

# Print some statistics
print("The duration of %i %% of the taxi trips was less than 35 minutes." % 
      (100 * np.where(trip_dur_mins < 35)[0].size / train['trip_duration'].size))
print("The median taxi trip duration was %i minutes." % trip_dur_mins.median())
print("The shortest taxi trip duration was %i second." % train['trip_duration'].min())
print("The longest taxi trip duration was %i days and %i hours." % (
    np.floor(trip_dur_mins.max() / (60 * 24)),
    np.floor(trip_dur_mins.max() / (60 * 24) % 1 * 24)))


# In[11]:


KMS_PER_RADIAN = 6371.0088
ROUND_DECIMALS = 4

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

map_osm = folium.Map(location=[40.730610, -73.935242], zoom_start=11, tiles='cartodbpositron')

for var_descr, var_names, color, icon in zip(
        ["Pickup location", "Drop-off location"],
        [["pickup_latitude", "pickup_longitude"], ["dropoff_latitude", "dropoff_longitude"]],
        ["orange", "black"], ["arrow-up", "arrow-down"]):
    print("Rounding to %i decimal places and removing duplicates from %s data." % (
        ROUND_DECIMALS, var_descr.lower()))
    train_copy = train.copy()  # Let's leave the original untouched
    train_copy[var_names] = train_copy[var_names].round(ROUND_DECIMALS)
    coords = train_copy[var_names].drop_duplicates()
    print("Kept %.1f %% of the %s data." % (100 * coords.shape[0] / train_copy.shape[0],
                                            var_descr.lower()))
    db = DBSCAN(eps=0.4 / KMS_PER_RADIAN, min_samples=0.0008*coords.shape[0], algorithm='ball_tree',
                metric='haversine', n_jobs=-1).fit(np.radians(coords.as_matrix()))
    # There is one cluster for noisy examples, labeled as -1 
    n_clusters = (len(set(db.labels_)) - 1 if -1 in db.labels_ else
                  len(set(db.labels_)))
    print('Found {} {} clusters.\n-----------------------------------'.format(
        n_clusters, var_descr.lower()))
    clusters_coords = [coords[db.labels_ == n] for n in range(n_clusters)]
    # Find the point in each cluster that is closest to its centroid
    centermost_points = [get_centermost_point(list(c.itertuples(index=False))) for
                         c in clusters_coords]
    clusters_with_all_orig_data = [pd.merge(train_copy, c, how='inner', on=var_names)
                                   for c in clusters_coords]
    for i, ((lat, lng), c) in enumerate(zip(centermost_points, clusters_with_all_orig_data)):
        
        popup = folium.Popup(folium.IFrame(html="""
        <head>
          <style> body {{ font-family: sans-serif, helvetica, arial; font-size: 14px; }} </style>
        </head>
        <body>
          <p><b>{var_name} cluster no.</b> {cluster_ix}</p>
          <p><b>Nr. of examples in cluster</b>: {n_examples:9,}</p>
          <p><b>Average number of passengers</b>: {avg_pass}</p>
          <p><b>Median trip duration (minutes)</b>: {avg_trip_dur}</p>
          <p><b>Most common {pickup_or_dropoff} hour</b>: {pickup_h}</p>
        </body>
        """.format(
            var_name=var_descr, cluster_ix=i + 1, n_examples=c.shape[0],
            avg_pass=np.round(c["passenger_count"].mean(), 1),
            avg_trip_dur=np.round((c["trip_duration"] / 60).median(), 1),
            pickup_or_dropoff=var_names[0].split('_')[0],
            pickup_h=np.bincount(np.array([dt.hour for dt in c[
                "%s_datetime" % var_names[0].split('_')[0]]])).argmax()),
            width=290, height=165))
        
        [folium.Marker([lat, lng], popup=popup, icon=folium.Icon(color=color, icon=icon)).add_to(map_osm)]
map_osm

