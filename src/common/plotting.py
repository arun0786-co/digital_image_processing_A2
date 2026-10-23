import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.paths import ROOT

FIGSIZE = (9, 6)
DPI = 180


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def image_grid(images, titles, cols, cell=3.2, vmin=0, vmax=255, cmap="gray"):
    """grid of greyscale images with fixed display range. returns (fig, axes)."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), images, titles):
        if image.ndim == 3:
            ax.imshow(image, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
    return fig, axes


def write_table(path, lines):
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"  saved {path.relative_to(ROOT)}")
