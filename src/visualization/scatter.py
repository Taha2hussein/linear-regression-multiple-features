from pandas import Series
from visualization.baseviualization import start_plot, finish
import matplotlib.pyplot as plt

def draw_scatter(feature: Series, target: Series, visuaization_name: str = "scatter"):
    """
    Draw scatter plot between feature and target
    """
    start_plot()
    plt.scatter(feature, target, alpha=0.6, s = 20)
    finish(feature.name, target.name, visuaization_name=visuaization_name)
  