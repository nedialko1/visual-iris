from pathlib import Path

# Get the path of the current script file
script_path = Path(__file__).resolve()
# Get the directory containing *this* script
script_dir = script_path.parent

# --------------------------------

from models_io.load_model_params import *

def compare_model_stats(pars_v1, pars_v2):
    # Define the relative paths to the subfolder and file
    pars_folder = "../../models_io/saved_model_pars"
    path_v1 = '/'.join((pars_folder, pars_v1)) + ".txt"
    path_v2 = '/'.join((pars_folder, pars_v2)) + ".txt"

    # Combine them to get the absolute path to the files
    path_v1 = script_dir / Path(path_v1)
    path_v2 = script_dir / Path(path_v2)

    state0 = load_model_params(path_v1)
    state1 = load_model_params(path_v2)
    
    plot_data = [] 
    
    print(f"{'Layer':<20} | {'Max Δ':<10} | {'% Change':<10}")
    print("-" * 45)

    for key in state0:
        if key not in state1: continue
        
        t0, t1 = state0[key], state1[key]
        diff = t1 - t0
        
        # Calculations
        max_d = torch.max(torch.abs(diff)).item()
        l2_d = torch.norm(diff).item()
        orig_norm = torch.norm(t0).item()
        pct_change = (l2_d / orig_norm * 100) if orig_norm != 0 else 0
        
        print(f"{key:<20} | {max_d:10.6f} | {pct_change:9.2f}%")
        
        # Store for Summary Bar Chart
        plot_data.append({'layer': key, 'pct': pct_change})
        
        # Generate Heatmap for this specific layer
        generate_delta_heatmap(key, diff)

    # Generate the global summary chart
    save_path = "../../images/tensors_heatmaps/model_drift.png"
    save_path = script_dir / Path(save_path)
    generate_drift_plot(save_path, plot_data)

    total_l2_sq = 0.0
    for key in state0:
            if key in state1:
                diff = state1[key] - state0[key]
                total_l2_sq += torch.norm(diff).item()**2

    global_drift = total_l2_sq ** 0.5

    save_path = "../../images/hinton_drift.png"
    save_path = script_dir / Path(save_path)
    generate_hinton_drift(save_path, state0, state1, global_drift_val=global_drift)

    save_path = "../../images/tensors_heatmaps/architecture_drift.png"
    save_path = script_dir / Path(save_path)
    generate_architecture_heatmap(save_path, state0, state1, global_drift_val=global_drift)


# ====================================================================

from etc_utils.viz_bloks.hinton_diagram import hinton

layers = ['fc1', 'fc2', 'out']

def generate_hinton_drift(save_path, states_v0, states_v1, global_drift_val=0.0):
    """
    Plots the ΔW and Δb for each layer following the y = W*u + b flow.
    """

    # 1. FIND GLOBAL MAX DELTA for color consistency
    all_deltas = []
    for layer in layers:
        all_deltas.append((states_v1[f'{layer}.weight'] - states_v0[f'{layer}.weight']).numpy())
        all_deltas.append((states_v1[f'{layer}.bias'] - states_v0[f'{layer}.bias']).numpy())
    
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    gs = fig.add_gridspec(3, 2, width_ratios=[5, 0.4], wspace=0.1, hspace=0.4)

    # Use the absolute maximum across the WHOLE model for the color limits
    v_limit = max(np.max(np.abs(d)) for d in all_deltas)

    fig.suptitle("Model Drift", fontsize=14, fontweight='bold', y=0.98)
    plt.title(f"Scale: $\pm${v_limit:.3f} | Total Drift (L2 norm): {global_drift_val:.4f}", y=1.05)
    plt.axis('off')

    for i, layer in enumerate(layers):
        dw = (states_v1[f'{layer}.weight'] - states_v0[f'{layer}.weight']).numpy()
        db = (states_v1[f'{layer}.bias'] - states_v0[f'{layer}.bias']).numpy().reshape(-1, 1)

        # weights as stored (out, in) - i.e. Columns = Input, Rows = Output.

        ax_w = fig.add_subplot(gs[i, 0])  # None #
        txt = f"$\Delta$ {layer}.weight {dw.shape}"
        hinton(dw, title=txt, max_weight=None, ax=ax_w)

        ax_w.set_xticks(np.arange(dw.shape[1]))
        ax_w.set_xlim(-1, dw.shape[1])
        ax_w.set_yticks(np.arange(dw.shape[0]))
        ax_w.set_ylim(-1,dw.shape[0])

        ax_b = fig.add_subplot(gs[i, 1])
        txt = "$\Delta$ b"
        hinton(db, title=txt, max_weight=None, ax=ax_b)

        ax_b.set_xticks([])
        ax_b.set_yticks(np.arange(db.shape[0]))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close()

# ====================================================================

COLOR_MAP = "coolwarm"

def generate_architecture_heatmap(save_path, states_v0, states_v1, global_drift_val=0.0):
    """
    Plots the ΔW and Δb for each layer following the y = W*u + b flow.
    Rows: fc1, fc2, out
    Columns: Weights (Matrix), Biases (Column Vector)
    """

    # 1. FIND GLOBAL MAX DELTA for color consistency
    all_deltas = []
    for layer in layers:
        all_deltas.append((states_v1[f'{layer}.weight'] - states_v0[f'{layer}.weight']).numpy())
        all_deltas.append((states_v1[f'{layer}.bias'] - states_v0[f'{layer}.bias']).numpy())

    # Use the absolute maximum across the WHOLE model for the color limits
    v_limit = max(np.max(np.abs(d)) for d in all_deltas)

    plt.style.use('seaborn-v0_8-white')
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    gs = fig.add_gridspec(3, 2, width_ratios=[5, 0.4], wspace=0.1, hspace=0.4)

    fig.suptitle("Model Drift Heatmap", fontsize=14, fontweight='bold', y=0.98)

    for i, layer in enumerate(layers):
        dw = (states_v1[f'{layer}.weight'] - states_v0[f'{layer}.weight']).numpy()
        db = (states_v1[f'{layer}.bias'] - states_v0[f'{layer}.bias']).numpy().reshape(-1, 1)

        # weights as stored (out, in) - i.e. Columns = Input, Rows = Output.

        ax_w = fig.add_subplot(gs[i, 0])  # None #

        # Red = Positive Change, Blue = Negative Change, Gray = ~No Change
        im_w = ax_w.imshow(dw, cmap=COLOR_MAP, aspect='auto', vmin=-v_limit, vmax=v_limit, interpolation='nearest')

        ax_w.set_title(f"$\Delta$ {layer}.weight {dw.shape}", loc='left', fontsize=11, fontweight='bold')
        ax_w.tick_params(labelsize=8)
        ax_w.set_xticks(np.arange(dw.shape[1]))
        ax_w.set_yticks(np.arange(dw.shape[0]))

        # 4. Plot Bias Strip
        ax_b = fig.add_subplot(gs[i, 1])
        im_b = ax_b.imshow(db, cmap=COLOR_MAP, aspect='auto', vmin=-v_limit, vmax=v_limit, interpolation='nearest')
        ax_b.set_title("$\Delta$ b", loc='left', fontsize=10)
        ax_b.tick_params(labelsize=8)
        ax_b.set_xticks([])
        ax_b.set_yticks(np.arange(db.shape[0]))

    # 5. Single Shared Colorbar for the whole figure
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im_w, cax=cax, label='Absolute Weight Change ($\Delta$)')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Congruent architectural plot generated: {save_path}")


# =====================================

import matplotlib.pyplot as plt
import numpy as np

def generate_drift_plot(save_path, stats_data):
    """
    Generates a horizontal bar chart of the % Change per layer.
    stats_data: List of dicts or tuples containing [{'layer': name, 'pct': value}, ...]
    """
    # Sort data for better visual flow (highest change at top)
    stats_data.sort(key=lambda x: x['pct'])
    
    layers = [item['layer'] for item in stats_data]
    pcts = [item['pct'] for item in stats_data]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0.3, 0.8, len(layers)))
    
    bars = plt.barh(layers, pcts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add a vertical line at 100% for context
    plt.axvline(x=100, color='red', linestyle='--', label='100% (Baseline Magnitude)')
    
    plt.xlabel('Percentage Change (%)', fontsize=12, fontweight='bold')
    plt.title('Model Weight Drift by Layer', fontsize=14, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(axis='x', linestyle=':', alpha=0.6)

    # Add text labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[INFO] Visualization saved to: {save_path}")
    plt.close()

# -----------------------------------------------------

def generate_delta_heatmap(key, tensor_diff, save_dir="../../images/tensors_heatmaps"):
    """
    Generates a heatmap of the absolute differences in a specific layer.
    """
    # Use absolute values to show magnitude of change
    data = torch.abs(tensor_diff).numpy()
    
    # If it's a 1D Bias, reshape it to 2D for a better visual strip
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
        # figsize = (len(data[0]) * 0.8, 2)
        data = data.T
        figsize = (2, 5)
    else:
        figsize = (7, 5)
        # figsize = (data.shape[1] * 1.2, data.shape[0] * 0.8)

    plt.figure(figsize=figsize, facecolor='white')
    im = plt.imshow(data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label=f"Abs $\Delta$")

    plt.title(f"Model Drift {key}")

    """
    tensor_suffix = "bias"

    if tensor_suffix in key:
        x_tag = "Outputs"
        y_tag = "Inputs"
    else:
    """
    y_tag = "Outputs"
    x_tag = "Inputs"

    plt.ylabel(y_tag)
    plt.xlabel(x_tag)
    ax = plt.gca()
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Add text annotations for small matrices
    if data.size < 100:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f'{data[i, j]:.2f}', 
                         ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    save_path = script_dir / Path(f"{save_dir}/heatmap_{key}.png")
    plt.savefig(save_path)
    # plt.show()
    plt.close()
