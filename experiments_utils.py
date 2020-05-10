import matplotlib
matplotlib.use('Agg')
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

FIGURE_FOLDER = 'figures/'
EXTENSION = '.png'


def google_values():
    raw_data = pd.read_csv('datasets/2020_04_23_google.csv')
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])

    return raw_data

GOOGLE_VALUES = google_values()

def color_palette(data, hue):
    n_colors = 1 if hue == None else len(data[hue].unique())
    return sns.color_palette("cubehelix", n_colors=n_colors)


def lineplot(data, name, x, y, y_label, x_label='', hue=None, y_lim=None, fig_size=(6,4), legend_pos='best', style=None, show_error=False, **kwargs):
    fig = plt.figure(figsize=fig_size)
    sns.set(style="white", color_codes=True, font_scale=1.5)

    palette = color_palette(data, hue)

    g = sns.lineplot(x=x, y=y, hue=hue, data=data, palette=palette, legend="full", style=style, **kwargs)

    fig.tight_layout()

    if not legend_pos:
        g.legend_.remove()
    elif hue:
        handles, labels = g.get_legend_handles_labels()
        plt.legend(loc='best', prop={'size': 15}, handles=handles[1:], labels=labels[1:])

    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y',useOffset=False)

    if y_lim != None and len(y_lim) == 2:
        plt.ylim(y_lim)

    if show_error:
        error_band(data, x, hue)

    plt.savefig('figures/' + name + EXTENSION, dpi=300, bbox_inches='tight')
    plt.close('all')

def error_band(data, x, hue):
    ax = plt.gca()
    
    valid_labels = data[hue].unique()

    for line in ax.lines:
        if not line.get_label() in valid_labels:
            continue

        x_values = data.loc[data[hue] == line.get_label()][x].values
        y_min = data.loc[data[hue] == line.get_label()]['Min'].values
        y_max = data.loc[data[hue] == line.get_label()]['Max'].values

        ax.fill_between(x_values, y_min, y_max, color=line.get_color(), alpha=0.2, linewidth=0.0)
        
