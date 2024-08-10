import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout
)
from PyQt5.QtCore import QProcess, QTimer, QTime

class ScriptRunner(QWidget):
    def __init__(self, script_path, display_output=False):
        super().__init__()
        self.script_path = script_path
        self.display_output = display_output
        self.process = QProcess(self)
        self.timer = QTimer(self)
        self.start_time = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        if self.display_output:
            self.output_console = QTextEdit(self)
            self.output_console.setReadOnly(True)
            layout.addWidget(self.output_console)

        self.time_label = QLabel("时间: 00:00:00", self)
        layout.addWidget(self.time_label)

        self.setLayout(layout)

        self.timer.timeout.connect(self.update_time)

    def start_script(self):
        if os.path.exists(self.script_path):
            self.process.setProgram("python")
            self.process.setArguments([self.script_path])
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            if self.display_output:
                self.process.readyReadStandardOutput.connect(self.handle_output)
            self.process.started.connect(self.start_timer)
            self.process.finished.connect(self.stop_timer)
            self.process.start()
        else:
            print(f"路径无效: {self.script_path}")

    def handle_output(self):
        output = self.process.readAllStandardOutput().data().decode('utf-8')
        if self.display_output:
            self.output_console.append(output)

    def start_timer(self):
        self.start_time = QTime.currentTime()
        self.timer.start(1000)  # 每秒更新一次

    def update_time(self):
        if self.start_time:
            elapsed_time = self.start_time.secsTo(QTime.currentTime())
            self.time_label.setText(f"时间: {QTime(0, 0).addSecs(elapsed_time).toString('hh:mm:ss')}")

    def stop_timer(self):
        self.timer.stop()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Python Script Runner")
        central_widget = QWidget()
        # layout = QHBoxLayout(central_widget)
        layout = QVBoxLayout(central_widget)

        # 设置绝对路径
        script1_path = r"py_api\neuracle_py_api\example_online_read_data.py"
        script2_path = r"demos\brainstim_demos\stim_demo.py"
        script3_path = r"demos\brainstim_demos\stim_demoo.py"

        self.script1_runner = ScriptRunner(script1_path, display_output=True)
        self.script2_runner = ScriptRunner(script2_path, display_output=True)
        self.script3_runner = ScriptRunner(script3_path, display_output=True)

        # 创建按钮并连接信号
        script1_button = QPushButton("EEG信号")
        script1_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; font-weight: bold;")
        script1_button.clicked.connect(self.script1_runner.start_script)

        script2_button = QPushButton("范式刺激")
        script2_button.setStyleSheet("background-color: #2196F3; color: white; font-size: 16px; font-weight: bold;")
        script2_button.clicked.connect(self.script2_runner.start_script)

        script3_button = QPushButton("机械臂输出")
        script3_button.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; font-weight: bold;")
        script3_button.clicked.connect(self.script3_runner.start_script)

        # 添加按钮和对应的 ScriptRunner 窗口到布局中
        layout.addWidget(script1_button)
        layout.addWidget(self.script1_runner)
        layout.addWidget(script2_button)
        layout.addWidget(self.script2_runner)
        layout.addWidget(script3_button)
        layout.addWidget(self.script3_runner)

        self.setCentralWidget(central_widget)
        self.resize(1000, 800)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
