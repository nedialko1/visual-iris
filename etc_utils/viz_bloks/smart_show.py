import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

# --------------------------------
from pathlib import Path

# 1. Get the path of the current script file
script_path = Path(__file__).resolve()
# 2. Get the directory containing the script
script_dir = script_path.parent

# --------------------------------

def smart_show(fgSaveFigures=False, selectedFigures=None, outFileNames=None):
    """
    Alternative to plt.show() that handles conditional saving and
    filtering of matplotlib/PyQt6-based figures.
    """
    # 1. Default Behavior: Just show and exit
    if not fgSaveFigures:
        plt.show()
        return

    # 2. Get all active managers (this works for both Qt6 and standard backends)
    # Managers are stored in a dict-like structure: {fignum: manager}
    managers = _pylab_helpers.Gcf.get_all_fig_managers()
    if not managers:
        print("No active figures found to save.")
        return

    # Helper to save with high-res settings
    def save_fig(fig, filename):
        # 3. Define the relative path to the subfolder and file
        relative_subfolder_path = Path(f"../../images/{filename}")
        # 4. Combine them to get the absolute path to the data file
        filename = script_dir / relative_subfolder_path

        # Using 300 DPI for "best resolution possible" as requested
        fig.savefig(f"{filename}.png", dpi=300)   # , bbox_inches='tight'
        print(f"Saved: {filename}.png")

    # Case A: Dictionary provided {fignum: filename}
    if isinstance(selectedFigures, dict) and selectedFigures:
        for fignum, filename in selectedFigures.items():
            manager = _pylab_helpers.Gcf.get_fig_manager(fignum)
            if manager:
                save_fig(manager.canvas.figure, filename)
            else:
                print(f"Warning: Figure {fignum} not found.")

    # Case B: List or String provided (Reverse chronological order)
    elif isinstance(selectedFigures, (list, str)) or isinstance(outFileNames, (list, str)):
        # Normalize to a list
        targets = selectedFigures if selectedFigures else outFileNames
        if isinstance(targets, str):
            targets = [targets]

        # Matplotlib managers are usually ordered by creation; we reverse them
        active_managers = list(reversed(managers))

        for i, filename in enumerate(targets):
            if i < len(active_managers):
                save_fig(active_managers[i].canvas.figure, filename)

    # Case C: Dictionary/List is empty - Save all with default names
    else:
        for manager in managers:
            fignum = manager.num
            save_fig(manager.canvas.figure, f"Fig_{fignum}")

    # Final display if needed (optional, depends if you want to see them after saving)
    plt.show()
