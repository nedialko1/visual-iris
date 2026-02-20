import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from etc_utils.data_harvest_io import load_uci_data
from etc_utils.viz_bloks import smart_show

# 1. Load the UCI Iris data
X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)
X_std = StandardScaler().fit_transform(X)

import hot_weights_4 as HW

lineColors = ['r', 'g','b']
for iFig in range(2):
# ======================================================
# 3. Pseudo-PC complementary 2D subspaces:
    iiThis2D = iFig + np.array(range(0,3,2), dtype=int)
    W = HW.W1[iiThis2D]
    bias = HW.b1[iiThis2D]
    print(f"{iFig}: W = {W}, bias = {bias}")

    X_pca = np.matmul(X_std,W.T) + bias

# 4. Visualization: 2D Projection and Density

    plt.figure(figsize=(8, 8))  ## (14, 6)

    # Subplot 1: Scatter with variety
    plt.subplot(1, 1, 1)
    ax = plt.gca()

    x1=X_pca[:, 0]
    num_pts = x1.size

    xx=np.unique(x1)
    nx1 = xx.size

    print(f"{iFig}: nx1 = {nx1}")
    xL = np.min(xx)
    xR = np.max(xx)
    xx = np.linspace(xL, xR, 2*nx1 )
    y2=X_pca[:, 1]
    yy=np.unique(y2)
    # print(f"yy = [{yy}]")
    yL = np.min(yy) - 3
    yR = np.max(yy) + 3
    print(f"yy \in ({yL},{yR})")
    ny2 = yy.size
    ny2 = 2*ny2
    yy = np.linspace(yL, yR, ny2 )

    # ---------------------------------------------------
    # Ouf! What a rocket science complexity just to get a scatter legend right!

    sns.scatterplot(x=x1, y=y2, hue=y, style=y_labels, palette='gist_rainbow', s=100,
                    ax=ax, legend='full')

    num_hue = np.unique(y).size

    h, l = ax.get_legend_handles_labels()  # Gets all handles (hue + style)

    hue_colors = [handle.get_markerfacecolor() for handle in h[1:num_hue+1]]

    # Map labels to their respective facecolors
    color_map = {label: handle.get_markerfacecolor()
                 for handle, label in zip(h[1:num_hue+1], l[1:num_hue+1])}
    num_hue_levels = num_hue + 1

    style_handles = h[num_hue_levels:]
    style_labels = l[num_hue_levels:]

    # You must know which color belongs to which style label
    i = -1
    for handle, label in zip(style_handles, style_labels):
        # Example: If 'label' exists in your color_map, apply it
        if label in color_map:
            handle.set_color(color_map[label])
        elif i>-1:
            print(f"*** {i}: {hue_colors[i]}")
            handle.set_markerfacecolor(hue_colors[i])
        i = i + 1

    # Create the legend using only hue handles
    ax.legend(style_handles, style_labels)

    # ---------------------------------------------------

    lines_info = HW.W2.shape
    m = lines_info[0]
    n = lines_info[1]
    ## m = 1
    for kLine in range(m):
        a = HW.W2[kLine,iiThis2D]
        b = HW.b2[kLine]
        print(f"{iFig}({iiThis2D}),{kLine}: a = {a}; b = {b}")
        kx = a[0]
        ky = a[1]
        # Divider-line equation:
        # kx*x + ky*y + b = 0
        if abs(ky) > 0.01 and abs(kx/ky) < 10:
            print(f"*** Type 1")
            x1 = xx
            y1 = -(kx*x1 + b)/ky
            if kLine == 1:
                y1 -= 0.1
            zL = np.min(y1)
            zR = np.max(y1)
            print(f"{iFig},{kLine}: y1 \in ({zL},{zR})")
            # Clip inside the visible figure space (of interest):
            ## jjValid = np.where((y1>=yL) & (y1<=yR))
            jjInValid = np.where((y1 < yL) | (y1 > yR))
            # y1[(y1<yL) | (y1>yR)] = np.nan
            y1 = np.delete(y1, jjInValid)
            x1 = np.delete(x1, jjInValid)
        elif abs(kx) > 0.01:
            print(f"*** Type 2")
            x0 = -b / kx
            x1 = x0 + np.zeros((ny2,))
            y1 = yy
        else:
            print(f"*** Type 0")
            x1 = 0
            y1 = 0

        plt.plot(x1, y1, lineColors[kLine] )

    plt.title('2D pseudo-PCA Projection: Variety Clusters')
    plt.xlabel(f'x1')
    plt.ylabel(f'y2')

    plt.tight_layout()

    # break
# ======================================================
# plt.show()
smart_show.smart_show(fgSaveFigures=True,
                      selectedFigures={1: "stats_iris_2D_take4_F1",
                                       2: "stats_iris_2D_take4_F2"})

