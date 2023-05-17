import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Read the CSV file into a DataFrame
data = pd.read_csv('results.txt', sep='\s+', header=None)
data.columns = ['method', 'nodes', 'vertices', 'edges', 'features', 'forcomp', 'forcomm', 'backcomp', 'backcomm']
data['forward'] = data['forcomp'] + data['forcomm']
data['backward'] = data['backcomp'] + data['backcomm']
data['total'] = data['forward'] + data['backward']
data['density'] = data['edges'] / (data['vertices'] * data['vertices'])
data['forratio'] = data['forcomm'] / data['forward']
data['backratio'] = data['backcomm'] / data['backward']
# pd.set_option('display.max_rows', None)
data.sort_values(by=['method', 'nodes', 'features'], inplace=True)


graphs = [(262144, 6871948),
          (262144, 687194767),
          (1048576, 109951163),]

for vertices,edges in graphs:
    for features in [16, 128]:

        sample = data[(data['edges'] == edges) & (data['vertices'] == vertices)]
        x = np.arange(len(sample['nodes'].unique())) * 4

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        width = 0.8
        for ax, pas in zip(axes, ['for', 'back']):
            offset = 0  
            for model in ['VA', 'GAT', 'AGNN']:
                if model == "VA":
                    col = "#990000"
                elif model == "GAT":
                    col = "#009900"
                else:
                    col = "#000099"
                black = "#000000"
                darkgrey = "#666666"
                current_sample = sample[(sample['method'] == model) & (sample['features'] == features)]
                ax.bar(x + offset, current_sample[f'{pas}comp'], width=width, color=col)
                p = ax.bar(x + offset, current_sample[f'{pas}comm'], width=width, bottom=current_sample[f'{pas}comp'], color=darkgrey)
                ax.bar_label(p, current_sample[f'{pas}ratio'].map(lambda x: f'{x * 100:.0f}%'), label_type='edge')
                offset += 1.1
            if pas == 'for':
                ax.set_title("Forward Pass")    
            else:
                ax.set_title("Backward Pass")
            ax.grid(which = "major")
            ax.grid(which = "minor", linewidth = 0.2)
            # Set x tick positions and labels
            ax.set_xticks(x + 1.1)
            ax.set_xticklabels(sample['nodes'].unique())

            ax.set_xlabel("Compute Nodes")
            ax.set_ylabel("Runtime [s]")
        
        
        plt.tight_layout()
        plt.savefig(f'plots/comm_ratio_{vertices}_{edges}_{features}.pdf')


# for global legend
import matplotlib.patches as mpatches

# Manually define the labels and colors for the legends
legend_labels = ['VA', 'GAT', 'AGNN', 'Communication']
legend_colors = ["#990000", "#009900", "#000099", "#666666"]

# Create the custom legend handles
legend_handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]

# Create a figure with multiple axes for legends
fig, axes = plt.subplots(1, 4, figsize=(8, 0.5))

# Iterate over the axes and add legends
for ax, handle in zip(axes, legend_handles):
    ax.axis('off')  # Remove axes and ticks
    ax.legend(handles=[handle], frameon=False, loc='center')

# Show the figure
plt.tight_layout()
plt.savefig(f'plots/comm_ratio_legend.pdf')