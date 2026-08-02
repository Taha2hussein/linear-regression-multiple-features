from pandas import  Series
import matplotlib.pyplot as plt
from visualization.baseviualization import start_plot, finish

def draw_histogram(feature: Series, visuaization_name: str = "histogram"):
    """
    Draw histogram for a specific column in the DataFrame.
    """
    start_plot()
    plt.hist(feature)
    finish(feature.name, show=False, visuaization_name=visuaization_name)