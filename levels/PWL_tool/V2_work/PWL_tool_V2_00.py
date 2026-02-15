import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --- Configuration & Styling ---
N_COMPONENTS = 3
COLORS = ['#FF4B4B', '#4BFF4B', '#4B4BFF'] # Iris classes
COLOR_FLD = (255, 255, 0)      # Yellow for Initial FLD
COLOR_USER = (0, 255, 255)     # Cyan for User segments
COLOR_ACTIVE = (255, 0, 0)     # Red for "In-progress"


class InteractiveSegment(pg.GraphItem):
    """A segment with draggable endpoints."""
    def __init__(self, pos, pen):
        self.dragPoint = None
        # pos is [ [x1, y1], [x2, y2] ]
        self.data = np.array(pos)
        adj = np.array([[0, 1]]) # Connect point 0 to 1
        pg.GraphItem.__init__(self)
        self.setData(pos=self.data, adj=adj, pen=pen, size=10, symbol='d')


    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            # Logic to delete could go here
            pass


class PWLPlot(pg.PlotWidget):
    def __init__(self, title, dims, X_pca, y, labels):
        super().__init__()
        self.setTitle(title)
        self.dims = dims # e.g., (0, 1) for PC1, PC2
        self.showGrid(x=True, y=True)
        
        # Plot Scatter Data
        for i in range(len(np.unique(y))):
            mask = (y == i)
            self.plot(X_pca[mask, dims[0]], X_pca[mask, dims[1]], 
                      pen=None, symbol='o', symbolBrush=COLORS[i], symbolSize=7)


        self.segments = []
        self.temp_line = None
        self.start_pt = None


    def mousePressEvent(self, ev):
        vb = self.getViewBox()
        pos = vb.mapSceneToView(ev.position())
        
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.start_pt is None:
                self.start_pt = [pos.x(), pos.y()]
            else:
                # Commit Segment
                seg = InteractiveSegment([[self.start_pt[0], self.start_pt[1]], 
                                          [pos.x(), pos.y()]], pen=COLOR_USER)
                self.addItem(seg)
                self.segments.append(seg)
                self.start_pt = None
                if self.temp_line:
                    self.removeItem(self.temp_line)
                    self.temp_line = None
        super().mousePressEvent(ev)


    def mouseMoveEvent(self, ev):
        if self.start_pt:
            vb = self.getViewBox()
            pos = vb.mapSceneToView(ev.position())
            if self.temp_line: self.removeItem(self.temp_line)
            self.temp_line = pg.PlotDataItem([self.start_pt[0], pos.x()], 
                                             [self.start_pt[1], pos.y()], 
                                             pen=pg.mkPen(COLOR_ACTIVE, style=Qt.PenStyle.DashLine))
            self.addItem(self.temp_line)
        super().mouseMoveEvent(ev)


class PWLDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PWL Model Designer - Systems Engineering Focus")
        self.setup_data()
        self.init_ui()
        self.init_fld_lines()


    def setup_data(self):
        # 1. Load Data
        iris = load_iris()
        self.X, self.y = iris.data, iris.target
        self.feature_names = iris.feature_names
        
        # 2. Scale and PCA
        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(self.X)
        self.pca = PCA(n_components=N_COMPONENTS)
        self.X_pca = self.pca.fit_transform(X_std)


    def init_ui(self):
        main_widget = QWidget()
        self.layout = QHBoxLayout(main_widget)
        
        # PC1 vs PC2
        self.p1 = PWLPlot("PC1 vs PC2", (0, 1), self.X_pca, self.y, self.feature_names)
        # PC1 vs PC3
        self.p2 = PWLPlot("PC1 vs PC3", (0, 2), self.X_pca, self.y, self.feature_names)
        
        self.layout.addWidget(self.p1)
        self.layout.addWidget(self.p2)
        self.setCentralWidget(main_widget)
        self.statusBar().showMessage("L-Click: Start/End Segment | Ctrl+S: Export NN Weights")


    def init_fld_lines(self):
        """Initializes the M-1 Fisher lines as starting points."""
        def get_fld_params(idx1, idx2):
            # Simplified FLD for 2 sets of classes
            m1 = np.mean(self.X_pca[np.isin(self.y, idx1)], axis=0)
            m2 = np.mean(self.X_pca[np.isin(self.y, idx2)], axis=0)
            # Calculate w, b for the specific subplots
            # ... (logic follows your provided FLD function)
            pass


    def export_nn_logic(self):
        """
        Layer 1: PCA Components (3x4 matrix)
        Layer 2: Combined 3D Planes from Subplots
        Layer 3: Decision Logic (Class Signalers)
        """
        W1 = self.pca.components_ # The loadings
        # Further logic to extract weights from drawn lines...
        print("Exporting Refined_2.py style weights...")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PWLDesigner()
    window.show()
    sys.exit(app.exec())
