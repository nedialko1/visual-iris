# visual_iris_UI.py = 
# from visual_iris_UI import * 

## --- global WEIGHTS, BIASES

import sys
from pathlib import Path
from etc_utils.data_harvest_io import load_uci_data

import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QGraphicsEllipseItem, QGraphicsLineItem, QVBoxLayout,
                             QHBoxLayout, QWidget, QSlider, QComboBox, QLabel, QProgressBar)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPen, QColor, QBrush, QPainter, QFont

# ===============================================================

from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QShortcut, QKeySequence

FG_DEBUG = False # True
# ===============================================================

# *** canvas_save widget definition:

def setup_interactive_save(canvas_widget):
    # Bind Ctrl+S to the save action
    shortcut = QShortcut(QKeySequence("Ctrl+S"), canvas_widget)

    def on_save_triggered():
        # 1. Open the file naming dialog
        # Returns a tuple: (selected_file_path, selected_filter)
        file_path, _ = QFileDialog.getSaveFileName(
            canvas_widget,
            "Save Canvas As",
            "",
            "PNG Files (*.png);;All Files (*)"
        )

        # 2. Proceed only if the user didn't cancel
        if file_path:
            do_print_canvas(file_path, canvas_widget)

    shortcut.activated.connect(on_save_triggered)

def do_print_canvas(file_path, canvas_widget):
    # Ensure correct extension
    if not file_path.lower().endswith(".png"):
        file_path += ".png"

    folder_path = Path(file_path).parent

    if not folder_path.is_dir():
        print(f"ERROR: The folder '{folder_path}' does not exist.")
        return

    # Capture and save at current screen resolution
    pixmap = canvas_widget.grab()
    pixmap.save(file_path, "PNG")
    print(f"Canvas saved to: {file_path}")

# --- UI LOGIC ---

class NodeItem(QGraphicsEllipseItem):
    def __init__(self, x, y, r=22, is_input=False, label=""):
        super().__init__(x - r, y - r, 2 * r, 2 * r)
        self.is_input = is_input
        self.label_text = label
        self.setBrush(QBrush(Qt.GlobalColor.white if is_input else Qt.GlobalColor.lightGray))
        self.activation = 0.0

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        painter.setPen(Qt.GlobalColor.black)
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        val = int(round(self.activation * 1000))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(val))
        if self.label_text:
            painter.drawText(int(self.rect().right() + 10), \
                int(self.rect().center().y() + 5), self.label_text)


class ArrowItem(QGraphicsLineItem):
    def __init__(self, start, end, weight, WEIGHTS):
        self.WEIGHTS = WEIGHTS
        super().__init__(start.x(), start.y(), end.x(), end.y())
        color = Qt.GlobalColor.green if weight > 0 else Qt.GlobalColor.red
        # Scale width relative to max weight
        abs_all = np.abs(np.concatenate([w.flatten() for w in self.WEIGHTS]))
        width = 1 + (abs(weight) / np.max(abs_all)) * 5
        self.setPen(QPen(QColor(color), int(width)))

from sklearn.preprocessing import StandardScaler

class VisualIrisApp(QMainWindow):
    def __init__(self, WEIGHTS, BIASES, FIG_MINOR):
        self.FIG_MINOR = FIG_MINOR
        self.FIG_VERSION = 0

        self.LAST_LAYER = 3
        self.slidersTags = ["Hidden 1", "Hidden 2", "Output"]
        if len(WEIGHTS) < 3:
            self.LAST_LAYER = 1
            self.slidersTags = ["LDA 1"]
            self.FIG_VERSION = 3
        elif WEIGHTS[0].shape[0]<4:
            self.FIG_VERSION = 2
        elif WEIGHTS[0].shape[0] < 8:
            self.FIG_VERSION = 1
        super().__init__()
        self.setWindowTitle("Visual Iris v1.7 - ReLU Systems Engineering Edition")

        if self.FIG_VERSION == 3:
            self.resize(1200, 650)
        elif self.FIG_VERSION == 0:
            self.resize(1200, 950)
        else:
            self.resize(1200, 650)

        self.WEIGHTS = WEIGHTS
        self.BIASES = BIASES

        try:
            X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)
            self.X_std = StandardScaler().fit_transform(X)
            self.y_labels = y_labels

        except FileNotFoundError:
            print("Error: iris_data.csv missing.")
            sys.exit(1)

        self.init_ui()
        self.draw_network()

        init_spec = self.y_labels.unique()[self.FIG_MINOR]
        self.update_base_data(init_spec)

    def init_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QVBoxLayout(container)

        self.combo = QComboBox()
        self.combo.addItems(self.y_labels.unique())
        self.combo.setCurrentIndex(self.FIG_MINOR)

        self.combo.currentTextChanged.connect(self.update_base_data)
        # Selects the first item in the list

        main_layout.addWidget(QLabel("<b>Input Species Baseline:</b>"))
        main_layout.addWidget(self.combo)

        if self.FIG_VERSION == 0:
            self.scene = QGraphicsScene(0, 0, 1000, 600)
        else:
            self.scene = QGraphicsScene(0, 0, 1000, 300)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        main_layout.addWidget(self.view)

        self.sliders, self.val_labels, self.range_labels = [], [], []
        s_row = QHBoxLayout()
        for i in self.slidersTags:
            v = QVBoxLayout()
            r_lbl = QLabel(f"{i} Thr")
            r_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v_lbl = QLabel("0")
            v_lbl.setStyleSheet("background-color: #222; color: #0F0; font-family: monospace; font-weight: bold;")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v_lbl.setFixedWidth(50)
            s = QSlider(Qt.Orientation.Horizontal)
            s.valueChanged.connect(self.recompute_cascade)

            v.addWidget(r_lbl)
            val_c = QHBoxLayout();
            val_c.addStretch();
            val_c.addWidget(v_lbl);
            val_c.addStretch()
            v.addLayout(val_c);
            v.addWidget(s)

            self.sliders.append(s);
            self.val_labels.append(v_lbl);
            self.range_labels.append(r_lbl)
            s_row.addLayout(v)
        main_layout.addLayout(s_row)

        self.prob_bars = []
        self.species_names = self.y_labels.unique().tolist()
        for name in self.species_names:
            h = QHBoxLayout()
            h.addWidget(QLabel(f"<b>{name:15}</b>"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            self.prob_bars.append(bar)
            h.addWidget(bar)
            main_layout.addLayout(h)

    def draw_network(self):
        self.nodes = []
        if self.FIG_VERSION == 0:
            sizes = [4, 8, 9, 3]
        elif self.FIG_VERSION == 1:
            sizes = [4, 4, 3, 3]
        elif self.FIG_VERSION == 2:
            sizes = [4, 3, 3, 3]
        else:
            sizes = [4, 3]

        n_layers = len(sizes)
        if FG_DEBUG:
            print(f"*** [layers(1..{n_layers})]: {sizes}")

        x_step, y_step = 250, 60
        for l_idx, size in enumerate(sizes):
            layer = []
            for n_idx in range(size):
                x = l_idx * x_step + 100
                y = n_idx * y_step + (max(sizes) - size) * y_step / 2 + 50
                # ==============================
                if FG_DEBUG:
                    hiddenNeuronTag = f"{l_idx}.{n_idx}"
                else:
                    hiddenNeuronTag = ""
                lbl = self.species_names[n_idx] if l_idx == self.LAST_LAYER else hiddenNeuronTag
                node = NodeItem(x, y, is_input=(l_idx == 0), label=lbl)
                self.scene.addItem(node);
                layer.append(node)
            self.nodes.append(layer)

        if FG_DEBUG:
            m_nodes = [len(layer) for layer in self.nodes]
            print(f"*** [layers(1..{n_layers})]: {m_nodes}")

        for l_idx in range(n_layers - 1):
            w_mat = self.WEIGHTS[l_idx]
            if FG_DEBUG:
                print(f"*** [{len(self.nodes[l_idx])}] :--> [{len(self.nodes[l_idx+1])}]")
                print(f"*** LAYER[{l_idx}]: {w_mat}")
            for i, src in enumerate(self.nodes[l_idx]):
                if FG_DEBUG:
                    print(f"--- SRC[{i}]: {src.label_text}")
                for j, tgt in enumerate(self.nodes[l_idx + 1]):
                    if FG_DEBUG:
                        print(f">>> TGT[{j}]: {tgt.label_text}")
                    if w_mat[j, i] != 0:
                        arrow = ArrowItem(src.rect().center(), tgt.rect().center(), \
                            w_mat[j, i], self.WEIGHTS)
                        arrow.setZValue(-1);
                        self.scene.addItem(arrow)

    def update_base_data(self, species):
        condition_mask = (self.y_labels == species)
        subset = self.X_std[condition_mask]
        self.base_input = subset.mean(axis=0)

        # Baseline ReLU Forward Pass
        acts = [self.base_input]
        curr = self.base_input

        print(f"[curr]={curr.shape}={curr}")
        """
        print(f"WEIGHTS: {self.WEIGHTS}")
        print(f"BIASES: {self.BIASES}")
        """
        cnt = 0
        for w, b in zip(self.WEIGHTS, self.BIASES):
            if FG_DEBUG:
                print(f"{cnt}. [w]={w.shape}, [b]={b.shape} <> [curr]={curr.shape}")
            curr = np.maximum(0, np.dot(w, curr) + b)
            acts.append(curr)
            cnt += 1
        for i in range(self.LAST_LAYER):
            layer_vals = acts[i + 1]
            mn = int(np.floor(np.min(layer_vals) * 100))
            mx = int(np.ceil(np.max(layer_vals) * 100))
            if mn == mx: mn, mx = max(0, mn - 1), mn + 1
            self.sliders[i].setRange(mn, mx)
            self.sliders[i].setValue(mn)
            self.range_labels[i].setText(f"L{i + 1} Range (100x): {mn}-{mx}")

        self.recompute_cascade()

    def recompute_cascade(self):
        current_acts = [self.base_input]
        for i in range(len(self.WEIGHTS)):
            self.val_labels[i].setText(str(self.sliders[i].value()))
            prev_out = current_acts[-1]

            # ReLU Cascade: f(x) = max(0, Wx + b)
            if np.all(prev_out == 0):
                raw = np.zeros(self.WEIGHTS[i].shape[0])
            else:
                raw = np.maximum(0, np.dot(self.WEIGHTS[i], prev_out) + self.BIASES[i])

            thr = self.sliders[i].value() / 100.0
            masked = np.where(raw > thr, raw, raw) ## 0.0
            current_acts.append(masked)

        for l_idx, layer_act in enumerate(current_acts):
            thr = self.sliders[l_idx - 1].value() / 100.0 if l_idx > 0 else -1.0
            for n_idx, val in enumerate(layer_act):
                node = self.nodes[l_idx][n_idx]
                node.activation = val
                if not node.is_input:
                    active = (val > thr and val > 0)
                    node.setBrush(QBrush(QColor("yellow") if active else Qt.GlobalColor.lightGray))

        # Probability Bar Logic
        final = current_acts[-1]
        if np.sum(final) == 0:
            probs = np.zeros(3)
        else:
            shift_x = final - np.max(final)
            exps = np.exp(shift_x)
            probs = exps / np.sum(exps)

        for j, bar in enumerate(self.prob_bars):
            val = int(round(probs[j] * 100))
            bar.setValue(val)
            if probs[j] == np.max(probs) and np.max(probs) > 0:
                bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")
        self.scene.update()