import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QMessageBox
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DraggableHandle(pg.ScatterPlotItem):
    def __init__(self, parent_seg, index):
        super().__init__(size=12, symbol='+', brush='r')
        self.parent_seg = parent_seg
        self.index = index


    def mouseDragEvent(self, ev):
        if not ev.isStart():
            pos = ev.pos()
            self.parent_seg.pts[self.index] = [pos.x(), pos.y()]
            self.setData(pos=self.parent_seg.pts[self.index:self.index+1])
            self.parent_seg.update_line()
        ev.accept()


class PWLSegment:
    def __init__(self, plot, p1, p2, color=(255, 255, 0)):
        self.plot = plot
        self.pts = np.array([p1, p2], dtype=float)
        self.line = pg.PlotDataItem(self.pts[:,0], self.pts[:,1], pen=pg.mkPen(color, width=2))
        self.plot.addItem(self.line)
        
        self.h1 = DraggableHandle(self, 0)
        self.h2 = DraggableHandle(self, 1)
        self.h1.setData(pos=self.pts[0:1])
        self.h2.setData(pos=self.pts[1:2])
        self.plot.addItem(self.h1)
        self.plot.addItem(self.h2)


    def update_line(self):
        self.line.setData(self.pts[:,0], self.pts[:,1])


    def get_standard_form(self):
        """Returns normalized w_a, w_b, b for w_a*x + w_b*y + b = 0"""
        x1, y1 = self.pts[0]; x2, y2 = self.pts[1]
        wa = y1 - y2
        wb = x2 - x1
        b = x1*y2 - x2*y1
        norm = np.sqrt(wa**2 + wb**2)
        return wa/norm, wb/norm, b/norm


class PWLDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PWL Tool - Iris Classifier Generator")
        self.setup_data()
        self.init_ui()
        self.init_fld_proper()


    def setup_data(self):
        iris = load_iris()
        self.X, self.y = iris.data, iris.target
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.pca = PCA(n_components=3)
        self.X_pca = self.pca.fit_transform(X_scaled)
        # PCA Components [3, 4] -> 3 PCs (rows), 4 features (cols)
        self.W1 = self.pca.components_ 


    def init_ui(self):
        container = QWidget(); layout = QHBoxLayout(container)
        self.p1 = PWLPlotWidget("PC1 vs PC2", (0, 1), self.X_pca, self.y)
        self.p2 = PWLPlotWidget("PC1 vs PC3", (0, 2), self.X_pca, self.y)
        layout.addWidget(self.p1); layout.addWidget(self.p2)
        self.setCentralWidget(container)


    def init_fld_proper(self):
        """M-1 FLD calculation using the full class covariance."""
        def get_fld_params(dims, c1, c2):
            X_sub = self.X_pca[:, dims]
            X1, X2 = X_sub[np.isin(self.y, c1)], X_sub[np.isin(self.y, c2)]
            m1, m2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
            sw = np.cov(X1.T) + np.cov(X2.T)
            w = np.linalg.solve(sw, (m1 - m2))
            b = -0.5 * np.dot(w, (m1 + m2))
            return w, b


        for plot in [self.p1, self.p2]:
            for pairs in [([0], [1, 2]), ([1], [2])]:
                w, b = get_fld_params(plot.dims, pairs[0], pairs[1])
                x_lims = np.array([np.min(self.X_pca[:, plot.dims[0]]), np.max(self.X_pca[:, plot.dims[0]])])
                y_lims = (-w[0]*x_lims - b) / w[1]
                plot.add_segment([x_lims[0], y_lims[0]], [x_lims[1], y_lims[1]])


    def keyPressEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_S:
            self.export_to_file()


    def export_to_file(self):
        # Layer 2: 3D Planes [2, 3]
        W2_rows = []
        b2_rows = []
        for s1, s2 in zip(self.p1.segments, self.p2.segments):
            wa1, wb1, bias1 = s1.get_standard_form()
            wa2, wb2, bias2 = s2.get_standard_form()
            # Fusion logic: Combine the two 2D lines into a 3D normal vector
            # PC1 is shared across both plots.
            w_3d = np.array([(wa1 + wa2)/2.0, wb1, wb2]) 
            b_3d = (bias1 + bias2)/2.0
            W2_rows.append(w_3d)
            b2_rows.append(b_3d)


        W2 = np.array(W2_rows) # Shape [2, 3]
        b2 = np.array(b2_rows) # Shape [2]


        # Layer 3: Signaling Logic [3, 2]
        # Class 0: Plane1 > 0, Plane2 don't care -> [10, 0]
        # Class 1: Plane1 < 0, Plane2 > 0       -> [-10, 10]
        # Class 2: Plane1 < 0, Plane2 < 0       -> [-10, -10]
        W3 = np.array([[15.0, 0.0], [-15.0, 15.0], [-15.0, -15.0]])
        b3 = np.array([0.0, -5.0, -15.0])


        with open("PWL_Tool_Refined_1.py", "w") as f:
            f.write("# Generated PWL Tool Refined Classifier\nimport numpy as np\n\n")
            f.write(f"W1 = np.array({self.W1.tolist()})\n")
            f.write(f"b1 = np.zeros({self.W1.shape[0]})\n\n")
            f.write(f"W2 = np.array({W2.tolist()})\n")
            f.write(f"b2 = np.array({b2.tolist()})\n\n")
            f.write(f"W3 = np.array({W3.tolist()})\n")
            f.write(f"b3 = np.array({b3.tolist()})\n")
        
        QMessageBox.information(self, "Success", "Exported to PWL_Tool_Refined_1.py")


class PWLPlotWidget(pg.PlotWidget):
    def __init__(self, title, dims, X_pca, y):
        super().__init__(title=title)
        self.dims = dims; self.segments = []
        colors = ['#FF4B4B', '#4BFF4B', '#4B4BFF']
        for i in range(3):
            mask = (y == i)
            self.plot(X_pca[mask, dims[0]], X_pca[mask, dims[1]], pen=None, 
                      symbol='o', symbolBrush=colors[i], symbolSize=6)
    def add_segment(self, p1, p2):
        self.segments.append(PWLSegment(self, p1, p2))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PWLDesigner(); window.show()
    sys.exit(app.exec())
