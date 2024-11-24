"""
The plot_graphs function in Python sets up a plot with a given title, x-axis label, and y-axis label using matplotlib. It rotates x-axis labels, positions the legend, adjusts the layout, saves the plot as a PNG file in a specified path, and displays the plot.
"""

from utils.imports import *
from .config import *


def plot_graphs(title, x_label, y_label,palette):   
    """
    Plot graphs with the given title, x-axis label, and y-axis label.

    Parameters:
    title (str): The title of the graph.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    """
    sns.set_palette(palette)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    # Use bbox_to_anchor with values less than  1 to keep it inside the graph
    plt.legend(loc='upper right', bbox_to_anchor=(0.91, 1.01))  
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH} {title.replace(' ', '_')}.png")
    plt.show()
