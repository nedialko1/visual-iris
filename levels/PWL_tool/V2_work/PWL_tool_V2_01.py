import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --- Constants ---
N_COMPONENTS = 3
COLORS = ['#FF4B4B', '#4BFF4B', '#4B4BFF']


class PWLPlot(pg.PlotWidget):
    def __init__(self, title, dims, X_pca, y):
        super().__init__()
        self.setTitle(title)
        self.dims = dims # e.g. (0, 1)
        self.X_pca = X_pca
        self.y = y
        self.showGrid(x=True, y=True)
        
        # Plot Data
        for i in range(3):
            mask = (y == i)
            self.plot(X_pca[mask, dims[0]], X_pca[mask, dims[1]], 
                      pen=None, symbol='o', symbolBrush=COLORS[i], symbolSize=7)


        # Interaction State
        self.segments = [] # List of PlotDataItem (lines)
        self.nodes = []    # List of PlotDataItem (draggable points)
        self.temp_line = None
        self.active_chain = []


    def add_segment_manually(self, p1, p2, color=(255, 255, 0)):
        """Draws a segment and creates internal tracking."""
        line = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], 
                                pen=pg.mkPen(color, width=2), symbol='d', symbolSize=10)
        self.addItem(line)
        self.segments.append(line)
        return line


    def mousePressEvent(self, ev):
        vb = self.getViewBox()
        pos = vb.mapSceneToView(ev.position())
        
        if ev.button() == Qt.MouseButton.LeftButton:
            self.active_chain.append([pos.x(), pos.y()])
            # Add visual node
            node = pg.PlotDataItem([pos.x()], [pos.y()], symbol='+', symbolPen=(250,0,0))
            self.addItem(node)
        
        elif ev.button() == Qt.MouseButton.RightButton:
            if len(self.active_chain) >= 2:
                self.commit_chain()
        super().mousePressEvent(ev)


    def commit_chain(self):
        # Logic to turn active_chain into permanent segments
        p1, p2 = self.active_chain[-2], self.active_chain[-1]
        self.add_segment_manually(p1, p2, color=(0, 255, 255))
        self.active_chain = []
        if self.temp_line:
            self.removeItem(self.temp_line)


class ManualClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PWL Designer - Systems Engineering Mode")
        self.setup_data_science_pipeline()
        self.init_ui()
        self.calculate_and_plot_fld()


    def setup_data_science_pipeline(self):
        # UCI Loading Simulation (using Iris as proxy)
        iris = load_iris()
        self.X, self.y = iris.data, iris.target
        self.labels = iris.target_names


        # Preprocessing
        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(self.X)
        
        self.pca = PCA(n_components=N_COMPONENTS)
        self.X_pca = self.pca.fit_transform(X_std)


    def init_ui(self):
        container = QWidget()
        self.layout = QHBoxLayout(container)
        self.p1 = PWLPlot("PC1 vs PC2", (0, 1), self.X_pca, self.y)
        self.p2 = PWLPlot("PC1 vs PC3", (0, 2), self.X_pca, self.y)
        self.layout.addWidget(self.p1)
        self.layout.addWidget(self.p2)
        self.setCentralWidget(container)


    def calculate_and_plot_fld(self):
        """Generates the M-1 FLD lines and projects them to subplots."""
        def get_line_2d(X_p, y_p, c1, c2, d_indices):
            # Extract only the 2 dimensions relevant to the plot
            X_sub = X_p[:, d_indices]
            m1 = np.mean(X_sub[np.isin(y_p, c1)], axis=0)
            m2 = np.mean(X_sub[np.isin(y_p, c2)], axis=0)
            sw = np.dot((X_sub[np.isin(y_p, c1)]-m1).T, (X_sub[np.isin(y_p, c1)]-m1)) + \
                 np.dot((X_sub[np.isin(y_p, c2)]-m2).T, (X_sub[np.isin(y_p, c2)]-m2))
            w = np.linalg.solve(sw, (m1 - m2))
            b = -0.5 * np.dot(w, (m1 + m2))
            return w, b


        for plot in [self.p1, self.p2]:
            # Line 1: Class 0 vs {1, 2}
            w, b = get_line_2d(self.X_pca, self.y, [0], [1, 2], plot.dims)
            self.draw_fld_on_plot(plot, w, b)
            # Line 2: Class 1 vs {2}
            w, b = get_line_2d(self.X_pca, self.y, [1], [2], plot.dims)
            self.draw_fld_on_plot(plot, w, b)


    def draw_fld_on_plot(self, plot, w, b):
        x_min, x_max = np.min(self.X_pca[:, plot.dims[0]]), np.max(self.X_pca[:, plot.dims[0]])
        # w[0]*x + w[1]*y + b = 0  => y = (-w[0]*x - b) / w[1]
        y_min = (-w[0]*x_min - b) / w[1]
        y_max = (-w[0]*x_max - b) / w[1]
        plot.add_segment_manually([x_min, y_min], [x_max, y_max])


    def keyPressEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_S:
            self.export_to_python()
        elif ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_U:
            print("Undo triggered") # Logic for popping from segment stack


    def export_to_python(self):
        # Implementation of Layer 1 (PCA), Layer 2 (User Boundaries), Layer 3 (Signaling)
        print("--- Exporting Refined_2.py ---")
        # Logic to extract w, b from plot.segments...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ManualClassifierGUI()
    gui.show()
    sys.exit(app.exec())
