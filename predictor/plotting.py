from pytz import timezone

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from predictor.models import Photo
from config import MEDIA_DIR

MAX_HEAT = 100
MIN_HEAT = 0


class Heatmap:
    def __init__(self):
        self.heatmap_data = None
        self.heatmap_plot = None

        self.get_heatmap_data()
        self.plot_heatmap()

    def get_heatmap_data(self):
        query = Photo.objects
        all_predictions = pd.DataFrame.from_records(query.values())
        all_predictions['datetime'] = all_predictions['datetime'].dt.tz_convert('US/Pacific')
        all_predictions['round_datetime'] = all_predictions['datetime'].dt.floor('15min')
        all_predictions['heat_value'] = np.nan
        all_predictions.loc[all_predictions['winner'] == 'Not out', 'heat_value'] = 0
        all_predictions.loc[all_predictions['winner'] == 'Just the bottom', 'heat_value'] = 50
        all_predictions.loc[all_predictions['winner'] == 'Only the tip', 'heat_value'] = 50
        all_predictions.loc[all_predictions['winner'] == 'All the way out', 'heat_value'] = 100
        heatmap_data = all_predictions[['round_datetime', 'heat_value']].drop_duplicates()
        heatmap_data = heatmap_data.set_index('round_datetime').resample('15min').first()
        heatmap_data.reset_index(inplace=True)
        heatmap_data['date'] = pd.to_datetime(heatmap_data['round_datetime']).dt.date
        heatmap_data['time'] = pd.to_datetime(heatmap_data['round_datetime']).dt.time
        self.heatmap_data = heatmap_data.set_index('round_datetime')

    def plot_heatmap(self):
        # fake_index = pd.date_range('2020-01-01', '2020-05-01', freq='15min',
        #                            tz='US/Pacific')
        # heatmap_data = pd.DataFrame(np.random.randint(0, 3, len(fake_index)) * 50,
        #                             index=fake_index,
        #                             columns=['heat_value'])
        # heatmap_data['date'] = heatmap_data.index.date
        # heatmap_data['time'] = heatmap_data.index.time
        # # remove above
        # heatmap_data = heatmap_data.iloc[:-1]
        heatmap_data = self.heatmap_data
        years = heatmap_data.index.year.unique()

        num_years = len(years)
        fig, axes = plt.subplots(num_years, 1,
                                 figsize=(12, num_years * 5))

        if num_years > 1:
            for i, year in enumerate(years):
                self.single_plot(heatmap_data, year, axes[i])
        else:
            year = years.to_list()[0]
            self.single_plot(heatmap_data, year, axes)

        legend_elements = [Patch(facecolor='tab:blue', edgecolor='w',
                                 label='All the way out'),
                           Patch(facecolor='lightsteelblue', edgecolor='w',
                                 label='Tip is out'),
                           Patch(facecolor='lightsteelblue', edgecolor='w',
                                 label='Bottom only'),
                           Patch(facecolor='tab:grey', edgecolor='w',
                                 label='Not out'),
                           Patch(facecolor='ghostwhite', edgecolor='tab:grey',
                                 label='No data :('),
                           ]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.04, 0.5 * num_years),
                   loc="center left", borderaxespad=0)
        fig.suptitle("Tahoma Visibility from Seattle Waterfront", fontsize=20)
        plt.tight_layout()
        plt.savefig(MEDIA_DIR / 'heatmap_plot.png')
        plt.close()

    @staticmethod
    def single_plot(data, year, ax):
        full_year = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:45:00',
                                  freq='15min', tz='US/Pacific')
        full_year = pd.DataFrame({'dummy': [np.nan] * len(full_year)},
                                 index=full_year)

        data = data[(data.index.year == year)]
        plot_data = full_year.join(data, how='outer').drop(columns=['dummy'])

        plot_data['date'] = plot_data.index.date
        plot_data['time'] = plot_data.index.time

        date = plot_data['date']
        first_day = pd.date_range(f'{year}-01-01', f'{year}-01-01 23:45', freq='15min')

        x = date.unique()
        y = mdates.date2num(first_day)

        heat = plot_data['heat_value'].values.reshape(24 * 4,
                                                      len(date.unique()),
                                                      order="F")

        colors = ['tab:grey', 'lightsteelblue', 'tab:blue']
        cmap = LinearSegmentedColormap.from_list('tahoma', colors, N=3)

        ax.pcolormesh(x, y, heat, cmap=cmap)
        ax.yaxis_date()
        y_format = mdates.DateFormatter('%H:%M')
        ax.yaxis.set_major_formatter(y_format)
        ax.yaxis.set_major_locator(mdates.HourLocator(interval=2))


if __name__ == '__main__':
    h = Heatmap()
