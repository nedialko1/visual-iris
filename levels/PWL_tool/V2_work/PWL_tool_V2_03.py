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
        if ev.isStart(): ev.accept()
        pos = ev.pos()
        # Update point data
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


    def get_params(self):
        # w_a*x + w_b*y + b = 0
        x1, y1 = self.pts[0]; x2, y2 = self.pts[1]
        wa = y1 - y2
        wb = x2 - x1
        bias = x1*y2 - x2*y1
        return wa, wb, bias


class PWLDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris PWL Designer - Export Mode")
        self.setup_data()
        self.init_ui()
        self.init_fld()


    def setup_data(self):
        iris = load_iris()
        self.X, self.y = iris.data, iris.target
        self.scaler = StandardScaler()
        self.X_pca = PCA(n_components=3).fit_transform(self.scaler.fit_transform(self.X))
        self.pca_loadings = PCA(n_components=3).fit(self.scaler.fit_transform(self.X)).components_


    def init_ui(self):
        container = QWidget(); layout = QHBoxLayout(container)
        self.p1 = PWLPlotWidget("PC1 vs PC2", (0, 1), self.X_pca, self.y)
        self.p2 = PWLPlotWidget("PC1 vs PC3", (0, 2), self.X_pca, self.y)
        layout.addWidget(self.p1); layout.addWidget(self.p2)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Ctrl+S to export Refined_2.py")


    def init_fld(self):
        # Initial M-1 planes logic
        for plot in [self.p1, self.p2]:
            for pair in [([0], [1, 2]), ([1], [2])]:
                x_pts = np.array([np.min(self.X_pca[:, plot.dims[0]]), np.max(self.X_pca[:, plot.dims[0]])])
                # Mock FLD calc for startup positioning
                y_pts = [0.5, -0.5] if plot.dims[1] == 1 else [-0.5, 0.5] 
                plot.add_segment([x_pts[0], y_pts[0]], [x_pts[1], y_pts[1]])


    def keyPressEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_S:
            self.export_refined_py()


    def export_refined_py(self):
        # 1. Layer 1: PCA Weights
        W1 = self.pca_loadings.T # [4, 3]


        # 2. Layer 2: 3D Planes
        W2_list, b2_list = [], []
        for s1, s2 in zip(self.p1.segments, self.p2.segments):
            wa1, wb1, bias1 = s1.get_params()
            wa2, wb2, bias2 = s2.get_params()
            # Fusion: Plane = w1*PC1 + w2*PC2 + w3*PC3 + b
            # PC1 is shared, PC2 comes from Plot 1, PC3 from Plot 2
            W2_list.append([ (wa1 + wa2)/2, wb1, wb2 ])
            b2_list.append((bias1 + bias2)/2)
        
        W2 = np.array(W2_list)
        b2 = np.array(b2_list)


        # 3. Layer 3: Combinatorial Logic (3x3 for Iris)
        # We use high weights (10.0) to simulate sharp ReLU activation gates
        W3 = np.array([
            [ 10.0,  0.0],  # Class 0: Plane 1 is Positive
            [-10.0,  10.0], # Class 1: Plane 1 is Negative, Plane 2 is Positive
            [-10.0, -10.0]  # Class 2: Both are Negative
        ])
        b3 = np.array([0.0, -5.0, -15.0]) # Bias to threshold the combinations


        with open("PWL_Tool_Refined_1.py", "w") as f:
            f.write("import numpy as np\n\n")
            f.write(f"# --- Layer 1: PCA Loadings ---\nW1 = {repr(W1)}\n\n")
            f.write(f"# --- Layer 2: User Defined Planes ---\nW2 = {repr(W2)}\nb2 = {repr(b2)}\n\n")
            f.write(f"# --- Layer 3: Signaling Logic ---\nW3 = {repr(W3)}\nb3 = {repr(b3)}\n")
        
        QMessageBox.information(self, "Export", "Refined_2.py has been generated.")


class PWLPlotWidget(pg.PlotWidget):
    def __init__(self, title, dims, X_pca, y):
        super().__init__(title=title)
        self.dims = dims; self.segments = []
        for i in range(3):
            mask = (y == i)
            self.plot(X_pca[mask, dims[0]], X_pca[mask, dims[1]], pen=None, 
                      symbol='o', symbolBrush=['r','g','b'][i], symbolSize=6)


    def add_segment(self, p1, p2):
        self.segments.append(PWLSegment(self, p1, p2))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PWLDesigner(); ex.show()
    sys.exit(app.exec())
