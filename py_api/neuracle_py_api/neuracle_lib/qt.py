# import sys
# import subprocess
# from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle('Qt Interface to Run Python Scripts')
#         self.setGeometry(800, 400, 800, 800)
#
#         # 创建按钮
#         self.run_script1_button = QPushButton('程序一', self)
#         self.run_script1_button.clicked.connect(self.run_script1)
#
#         self.run_script2_button = QPushButton('程序二', self)
#         self.run_script2_button.clicked.connect(self.run_script2)
#
#         # 布局
#         layout = QVBoxLayout()
#         layout.addWidget(self.run_script1_button)
#         layout.addWidget(self.run_script2_button)
#
#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)
#
#     def run_script1(self):
#         subprocess.Popen(['python', 'D:/脑控机械臂/MetaBCI/MetaBCI/demos/brainstim_demos/stim_demo.py'])
#
#     def run_script2(self):
#         subprocess.Popen(['python', 'D:/脑控机械臂/MetaBCI/MetaBCI/py_api/neuracle_py_api/example_online_read_data.py'])
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
import sys
import subprocess
import time
import psutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QTime


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # 用于存储每个脚本的开始时间
        self.start_times = {'script1': None, 'script2': None}
        # 用于存储每个脚本的定时器
        self.timers = {'script1': None, 'script2': None}

    def initUI(self):
        self.setWindowTitle('Run Scripts and Show Time')
        self.setGeometry(1120, 720, 400, 200)

        # 创建标签来显示每个脚本的运行时间
        self.timeLabel1 = QLabel('Script 1 Time: 00:00:00', self)
        self.timeLabel2 = QLabel('Script 2 Time: 00:00:00', self)

        # 创建按钮
        self.run_script1_button = QPushButton('脑电采集', self)
        self.run_script1_button.clicked.connect(self.run_script1)
        self.run_script1_button.setStyleSheet("background-color: lightblue; font-size: 16pt;")

        self.run_script2_button = QPushButton('范式刺激', self)
        self.run_script2_button.clicked.connect(self.run_script2)
        self.run_script2_button.setStyleSheet("background-color: lightgreen; font-size: 16pt;")

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.timeLabel1)
        layout.addWidget(self.timeLabel2)
        layout.addWidget(self.run_script1_button)
        layout.addWidget(self.run_script2_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 设置定时器，每秒更新一次时间显示
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTimes)
        self.timer.start(10)  # 每1000毫秒（1秒）触发一次

    def updateTimes(self):
        current_time = QTime.currentTime()
        for script in ['script1', 'script2']:
            if self.start_times[script] is not None:
                elapsed = self.start_times[script].secsTo(current_time)  # 计算经过的秒数
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                if script == 'script1':
                    self.timeLabel1.setText(f'Script 1 Time: {hours:02}:{minutes:02}:{seconds:02}')
                else:
                    self.timeLabel2.setText(f'Script 2 Time: {hours:02}:{minutes:02}:{seconds:02}')

    def run_script1(self):
        if self.timers['script1'] is not None:
            return  # 避免重复启动同一个脚本

        self.start_times['script1'] = QTime.currentTime()  # 记录开始时间
        self.timers['script1'] = subprocess.Popen(['python', 'D:/脑控机械臂/MetaBCI/MetaBCI/py_api/neuracle_py_api/example_online_read_data.py'])

    def run_script2(self):
        if self.timers['script2'] is not None:
            return  # 避免重复启动同一个脚本

        self.start_times['script2'] = QTime.currentTime()  # 记录开始时间
        self.timers['script2'] = subprocess.Popen(['python', 'D:/脑控机械臂/MetaBCI/MetaBCI/demos/brainstim_demos/stim_demo.py'])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())