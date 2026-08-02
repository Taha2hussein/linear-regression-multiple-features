from pathlib import Path
import matplotlib.pyplot as plt


def save_plot(path: str) -> None:
    """
    Save current matplotlib figure.
    """

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    
def finish_plot(save_path=None, show=True):
    """
    Finish the current plot by saving and/or showing it.
    """
    if save_path:
        save_plot(save_path)
    if show:
        plt.show()
    plt.close()