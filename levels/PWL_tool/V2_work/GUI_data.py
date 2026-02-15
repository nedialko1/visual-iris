import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from sklearn.datasets import load_iris


class PWLPlot(pg.PlotWidget):
    def __init__(self, title, feature_indices, df, parent_gui):
        super().__init__()
        self.setTitle(title)
        self.parent_gui = parent_gui
        self.setMouseEnabled(x=False, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.showGrid(x=True, y=True)

        # This helps the widget grab attention on the first click
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.active_pts = []
        self.active_items = []
        self.temp_line = None

        idx_x, idx_y = feature_indices
        colors = ['#FF4B4B', '#4BFF4B', '#4B4BFF']
        for t in range(3):
            mask = df['species'] == t
            self.plot(df.iloc[:, idx_x][mask].values, df.iloc[:, idx_y][mask].values,
                      pen=None, symbol='o', symbolBrush=colors[t], symbolSize=7)

    def mousePressEvent(self, ev):
        # Immediate focus on press
        self.setFocus()
        vb = self.getPlotItem().vb
        data_pt = vb.mapSceneToView(ev.position())
        x, y = data_pt.x(), data_pt.y()

        if ev.button() == Qt.MouseButton.LeftButton:
            if not self.active_pts:
                self.add_node(x, y, is_active=True)

        elif ev.button() == Qt.MouseButton.RightButton:
            if self.active_pts:
                self.commit_segment(x, y)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self.active_pts:
            vb = self.getPlotItem().vb
            data_pt = vb.mapSceneToView(ev.position())
            start = self.active_pts[-1]
            if self.temp_line:
                self.removeItem(self.temp_line)
            self.temp_line = pg.PlotDataItem([start[0], data_pt.x()], [start[1], data_pt.y()],
                                             pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
            self.addItem(self.temp_line)
        super().mouseMoveEvent(ev)

    def add_node(self, x, y, is_active=False):
        self.active_pts.append([x, y])
        symbol = "+" if is_active else "d"
        color = 'r' if is_active else (150, 150, 150)
        pt = pg.PlotDataItem([x], [y], symbol=symbol, symbolSize=12, symbolPen=color, symbolBrush=color)
        self.addItem(pt)
        self.active_items.append(pt)

    def commit_segment(self, x, y):
        # Convert the 'current' red cross to gray
        for item in self.active_items:
            if isinstance(item, pg.PlotDataItem) and item.opts.get('symbol') == "+":
                item.setSymbol("d")
                item.setSymbolBrush((150, 150, 150))
                item.setSymbolPen((150, 150, 150))

        # Add the line
        prev_x, prev_y = self.active_pts[-1]
        line = pg.PlotDataItem([prev_x, x], [prev_y, y], pen=pg.mkPen(150, 150, 150, width=1))
        self.addItem(line)
        self.active_items.append(line)

        # Add new active node
        self.add_node(x, y, is_active=True)

    def finalize(self):
        # Guard: Only finalize if a path has been started (at least 2 points)
        if len(self.active_pts) < 2:
            return None

        self.addItem(pg.PlotDataItem([self.active_pts[0][0]], [self.active_pts[0][1]],
                                     symbol='p', symbolBrush=(100, 100, 100), symbolSize=15))
        self.addItem(pg.PlotDataItem([self.active_pts[-1][0]], [self.active_pts[-1][1]],
                                     symbol='h', symbolBrush=(100, 100, 100), symbolSize=15))

        for item in self.active_items:
            if isinstance(item, pg.PlotDataItem) and item.opts.get('symbol') == "+":
                item.setSymbol("d")
                item.setSymbolBrush((150, 150, 150))

        if self.temp_line:
            self.removeItem(self.temp_line)
            self.temp_line = None

        chain = list(self.active_pts)
        self.active_pts = []
        self.active_items = []
        return chain


class ManualClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PWL Designer - Focus Refined")
        self.resize(1200, 600)

        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['species'] = iris.target

        container = QWidget()
        self.setCentralWidget(container)
        self.layout = QHBoxLayout(container)

        self.p1 = PWLPlot("Sepal Space", (0, 1), self.df, self)
        self.p2 = PWLPlot("Petal Space", (2, 3), self.df, self)
        self.plots = [self.p1, self.p2]

        for p in self.plots:
            self.layout.addWidget(p)

        self.finalized_data = {self.p1: [], self.p2: []}

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            # Targeted finalization based on focus/hover
            for p in self.plots:
                if p.hasFocus() or p.underMouse():
                    chain = p.finalize()
                    if chain:
                        self.finalized_data[p].append(chain)
        elif ev.modifiers() & Qt.KeyboardModifier.ControlModifier and ev.key() == Qt.Key.Key_S:
            self.export_logic()

    def export_logic(self):
        print("\n--- Final Export ---")
        # Logic for W, b remains the same
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ManualClassifierGUI()
    gui.show()
    sys.exit(app.exec())