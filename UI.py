import sys
import os
import cv2
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView,
    QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

from ultralytics import YOLO


class DarkDetectionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO("best.pt")
        self.image = None
        self.result = None
        self.img_path = ""

        self.progress = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        self.init_ui()

    # ================= UI =================
    def init_ui(self):
        self.setWindowTitle("钢材缺陷检测平台")
        self.resize(1250, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(15)

        # ---------- 标题 ----------
        title = QLabel("钢材缺陷检测平台")
        title.setAlignment(Qt.AlignCenter)
        title.setFixedHeight(55)
        title.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #e6e6e6;
                background-color: #1f1f1f;
                border-radius: 10px;
            }
        """)
        main.addWidget(title)

        # ---------- 图像区 ----------
        img_layout = QHBoxLayout()
        img_layout.setSpacing(15)

        self.left_img = QLabel("原始图像")
        self.right_img = QLabel("检测结果")

        for lab in (self.left_img, self.right_img):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(520, 360)
            lab.setStyleSheet("""
                QLabel {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 12px;
                    color: #aaaaaa;
                    font-size: 16px;
                }
            """)

        img_layout.addWidget(self.left_img)
        img_layout.addWidget(self.right_img)
        main.addLayout(img_layout)

        # ---------- 进度 ----------
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: #1f1f1f;
                border-radius: 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)

        progress_label = QLabel("检测进度")
        progress_label.setStyleSheet("color:#cfcfcf;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2a2a2a;
                border-radius: 8px;
                color: #cccccc;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3a86ff;
                border-radius: 8px;
            }
        """)

        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        main.addWidget(progress_frame)

        # ---------- 表格 ----------
        table_label = QLabel("检测结果与位置信息")
        table_label.setStyleSheet("color:#e0e0e0; font-size:16px;")
        main.addWidget(table_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["目标ID", "文件路径", "类别", "置信度", "坐标位置"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                color: #dddddd;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #1f1f1f;
                color: #dddddd;
                padding: 6px;
                border: none;
            }
        """)
        main.addWidget(self.table)

        # ---------- 按钮 ----------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(25)  #

        self.open_btn = QPushButton("📁 选择图片")
        self.start_btn = QPushButton("▶ 开始检测")
        self.exit_btn = QPushButton("退出")

        for btn in (self.open_btn, self.start_btn, self.exit_btn):
            btn.setFixedHeight(40)
            btn.setMinimumWidth(140)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a86ff;
                    color: white;
                    border-radius: 10px;
                    font-size: 14px;
                    padding: 6px 18px;
                }
                QPushButton:hover {
                    background-color: #4d96ff;
                }
                QPushButton:disabled {
                    background-color: #555555;
                    color: #aaaaaa;
                }
            """)

        self.start_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_image)
        self.start_btn.clicked.connect(self.detect)
        self.exit_btn.clicked.connect(self.close)

        btn_layout.addStretch()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addStretch()

        main.addLayout(btn_layout)

    # ================= 选择图片 =================
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(Path.home()),
            "Images (*.jpg *.png *.jpeg)"
        )
        if not path:
            return

        self.image = cv2.imread(path)
        self.img_path = path
        self.show_image(self.image, self.left_img)
        self.start_btn.setEnabled(True)

    # ================= 推理 =================
    def detect(self):
        self.progress = 0
        self.progress_bar.setValue(0)
        self.timer.start(40)

        results = self.model(self.image)
        self.result = results[0]

        self.timer.stop()
        self.progress_bar.setValue(100)

        img = self.result.plot()
        self.show_image(img, self.right_img)
        self.fill_table()

    # ================= 表格 =================
    def fill_table(self):
        self.table.setRowCount(0)
        if self.result.boxes is None:
            return

        for i, box in enumerate(self.result.boxes):
            row = self.table.rowCount()
            self.table.insertRow(row)

            cls = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()

            self.table.setItem(row, 0, QTableWidgetItem(str(i)))
            self.table.setItem(row, 1, QTableWidgetItem(self.img_path))
            self.table.setItem(row, 2, QTableWidgetItem(self.model.names[cls]))
            self.table.setItem(row, 3, QTableWidgetItem(f"{conf*100:.2f}%"))
            self.table.setItem(row, 4, QTableWidgetItem(str([int(x) for x in xyxy])))

    # ================= 进度 =================
    def update_progress(self):
        if self.progress < 90:
            self.progress += 2
            self.progress_bar.setValue(self.progress)

    # ================= 显示图像 =================
    def show_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)
        label.setText("")


# ================= 主入口 =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            background-color: #121212;
            font-family: "Microsoft YaHei";
        }
    """)
    window = DarkDetectionSystem()
    window.show()
    sys.exit(app.exec_())
