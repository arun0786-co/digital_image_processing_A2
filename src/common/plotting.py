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


def panel_grid(count, cols=2, width=3.6, height=3.6, rows=None):
    """figure with count panels of width x height inches, cols to a row. rows can list
    how many panels go in each row instead; a row with fewer panels is centred.
    returns (fig, list of axes in reading order)."""
    if rows is None:
        rows = [cols] * (count // cols) + ([count % cols] if count % cols else [])
    fig = plt.figure(figsize=(width * cols, height * len(rows)))
    grid = fig.add_gridspec(len(rows), 2 * cols)
    axes = []
    for row, in_row in enumerate(rows):
        start = cols - in_row
        for col in range(in_row):
            axes.append(fig.add_subplot(grid[row, start + 2 * col:start + 2 * col + 2]))
    return fig, axes


def image_grid(images, titles, cols=2, cell=3.6, aspect=1.0, vmin=0, vmax=255, cmap="gray"):
    """images with a fixed display range, cols to a row, each cell cell x (aspect cell)
    inches. returns (fig, list of axes)."""
    fig, axes = panel_grid(len(images), cols, cell, cell * aspect)
    for ax, image, title in zip(axes, images, titles):
        if image.ndim == 3:
            ax.imshow(image, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    return fig, axes


def write_table(path, lines):
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"  saved {path.relative_to(ROOT)}")
