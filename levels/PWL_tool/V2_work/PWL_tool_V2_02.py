import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QPointF
import pyqtgraph as pg
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --- Geometric Constants & Colors ---
COLOR_FLD = (255, 255, 0)      # Yellow for FLD
COLOR_USER = (0, 255, 255)     # Cyan for Manual
COLOR_NODE = (255, 0, 0)       # Red for Handles


class DraggableNode(pg.GraphItem):
    """A vertex handle that updates its parent line when moved."""
    def __init__(self, parent_segment, index):
        self.parent_segment = parent_segment
        self.index = index
        pg.GraphItem.__init__(self)
        self.setData(pos=np.array([parent_segment.pts[index]]), size=12, symbol='+', brush=COLOR_NODE)


    def mouseDragEvent(self, ev):
        if ev.isStart():
            self.startPos = self.data['pos'][0]
        
        delta = ev.pos() - ev.lastPos()
        self.data['pos'][0] += [delta.x(), delta.y()]
        
        # Update the parent line's data
        self.parent_segment.pts[self.index] = self.data['pos'][0]
        self.parent_segment.update_line()
        self.update()
        ev.accept()


class PWLSegment:
    """Logical container for a line and its two draggable handles."""
    def __init__(self, plot_widget, p1, p2, color=COLOR_FLD):
        self.plot = plot_widget
        self.pts = np.array([p1, p2])
        self.line = pg.PlotDataItem(self.pts[:,0], self.pts[:,1], pen=pg.mkPen(color, width=2))
        self.plot.addItem(self.line)
        
        self.h1 = DraggableNode(self, 0)
        self.h2 = DraggableNode(self, 1)
        self.plot.addItem(self.h1)
        self.plot.addItem(self.h2)


    def update_line(self):
        self.line.setData(self.pts[:,0], self.pts[:,1])


    def get_equation(self):
        """Returns (w_a, w_b, bias) such that w_a*x + w_b*y + b = 0"""
        x1, y1 = self.pts[0]
        x2, y2 = self.pts[1]
        # Standard form from two points: (y1 – y2)x + (x2 – x1)y + (x1y2 – x2y1) = 0
        wa = y1 - y2
        wb = x2 - x1
        bias = x1*y2 - x2*y1
        # Normalize
        norm = np.sqrt(wa**2 + wb**2)
        return wa/norm, wb/norm, bias/norm


class PWLPlot(pg.PlotWidget):
    def __init__(self, title, dims, X_pca, y):
        super().__init__()
        self.setTitle(title)
        self.dims = dims
        self.X_pca = X_pca
        self.segments = []
        
        # Plot Scatter
        for i in range(3):
            mask = (y == i)
            self.plot(X_pca[mask, dims[0]], X_pca[mask, dims[1]], 
                      pen=None, symbol='o', symbolBrush=['#FF4B4B','#4BFF4B','#4B4BFF'][i], symbolSize=7)


    def add_segment(self, p1, p2, color=COLOR_FLD):
        seg = PWLSegment(self, p1, p2, color)
        self.segments.append(seg)


class PWLDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_data()
        self.init_ui()
        self.init_fld()


    def setup_data(self):
        iris = load_iris()
        self.X_std = StandardScaler().fit_transform(iris.data)
        self.pca = PCA(n_components=3)
        self.X_pca = self.pca.fit_transform(self.X_std)
        self.y = iris.target


    def init_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        self.p1 = PWLPlot("PC1 vs PC2", (0, 1), self.X_pca, self.y)
        self.p2 = PWLPlot("PC1 vs PC3", (0, 2), self.X_pca, self.y)
        layout.addWidget(self.p1)
        layout.addWidget(self.p2)
        self.setCentralWidget(container)


    def init_fld(self):
        """Mandate: Generate M-1 FLD lines per plot."""
        for plot in [self.p1, self.p2]:
            for pair in [([0], [1, 2]), ([1], [2])]:
                w, b = self.get_fld_line(plot.dims, pair[0], pair[1])
                # Project line to plot boundaries
                x_pts = np.array([np.min(self.X_pca[:, plot.dims[0]]), np.max(self.X_pca[:, plot.dims[0]])])
                y_pts = (-w[0]*x_pts - b) / w[1]
                plot.add_segment([x_pts[0], y_pts[0]], [x_pts[1], y_pts[1]])


    def get_fld_line(self, dims, c1, c2):
        X_sub = self.X_pca[:, dims]
        m1, m2 = np.mean(X_sub[np.isin(self.y, c1)], axis=0), np.mean(X_sub[np.isin(self.y, c2)], axis=0)
        sw = np.cov(X_sub[np.isin(self.y, c1)].T) + np.cov(X_sub[np.isin(self.y, c2)].T)
        w = np.linalg.solve(sw, (m1 - m2))
        return w, -0.5 * np.dot(w, (m1 + m2))


    def keyPressEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_S:
            self.export_nn()


    def export_nn(self):
        """Fuses 2D segments into 3D Weights (W2)."""
        print("\n--- Generating Neural Network Weights ---")
        W2, b2 = [], []
        
        # Iterate through paired segments
        for s1, s2 in zip(self.p1.segments, self.p2.segments):
            eq1 = s1.get_equation() # (w1_pc1, w1_pc2, b1)
            eq2 = s2.get_equation() # (w2_pc1, w2_pc3, b2)
            
            # Combine into 3D plane: w_pc1*PC1 + w_pc2*PC2 + w_pc3*PC3 + b = 0
            # We average the PC1 components and bias from both views for a unified plane
            w_3d = [ (eq1[0] + eq2[0])/2, eq1[1], eq2[1] ]
            bias = (eq1[2] + eq2[2])/2
            W2.append(w_3d)
            b2.append(bias)


        # Output formatting (Refined_2.py style)
        print(f"W1 (PCA Loadings):\n{self.pca.components_}")
        print(f"W2 (User Planes):\n{np.array(W2)}")
        print(f"b2 (Biases):\n{np.array(b2)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PWLDesigner()
    window.show()
    sys.exit(app.exec())
