from pandas import Series
from visualization.baseviualization import start_plot, finish
import matplotlib.pyplot as plt

def draw_box_plot(feature: Series, visuaization_name: str = "boxplot"):
    """
    Draw boxplot for a specific column in the DataFrame.
    """
    start_plot()
    plt.boxplot(feature)
    finish(feature.name, show=False, visuaization_name=visuaization_name)