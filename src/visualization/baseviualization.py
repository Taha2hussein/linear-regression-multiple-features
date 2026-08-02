import matplotlib.pyplot as plt
from utils.plot import finish_plot 
from config.config import PLOTS_DIR


def start_plot():
    """
    Start a new matplotlib figure.
    """
    plt.figure(figsize=(8,5))
    
def finish(feature_name: str = "", target_name: str = "", show: bool = False, visuaization_name: str = ""):
    """
   Finalize the current matplotlib plot.
    """
    filename = (
        f"{feature_name}_vs_{target_name}"
        .replace("/", "_")
        .replace(" ", "_")
    )
    plt.xlabel(f"{feature_name}")
    plt.ylabel(f"{target_name}")
    plt.title(f"{feature_name} vs {target_name}")
    plt.grid(True)
    plt.tight_layout()
    finish_plot(save_path=f"{PLOTS_DIR}/{visuaization_name}/{filename}.png", show=False)
  