import sys
import subprocess
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QTime, QThread, pyqtSignal


class ScriptRunner(QThread):
    timeUpdated = pyqtSignal(str, str)

    def __init__(self, script_path, parent=None):
        super().__init__(parent)
        self.script_path = script_path
        self.start_time = None
        self.process = None

    def run(self):
        # Record the start time
        self.start_time = QTime.currentTime()
        self.process = subprocess.Popen(['python', self.script_path], shell=True)
        self.process.wait()  # Wait for the process to finish

    def get_elapsed_time(self):
        if self.start_time:
            current_time = QTime.currentTime()
            elapsed = self.start_time.secsTo(current_time)  # Calculate elapsed time in seconds
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            return f'{hours:02}:{minutes:02}:{seconds:02}'
        return '00:00:00'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize ScriptRunner threads
        self.script1_runner = ScriptRunner('D:/MetaBCI/MetaBCI/py_api/neuracle_py_api/example_online_read_data.py')
        self.script2_runner = ScriptRunner('D:/MetaBCI/MetaBCI/demos/brainstim_demos/stim_demo.py')

        # Connect signals to update time labels
        self.script1_runner.timeUpdated.connect(self.update_time1)
        self.script2_runner.timeUpdated.connect(self.update_time2)

    def initUI(self):
        self.setWindowTitle('Run Scripts and Show Time')
        self.setGeometry(100, 100, 400, 200)

        # Create labels to show each script's running time
        self.timeLabel1 = QLabel('Script 1 Time: 00:00:00', self)
        self.timeLabel2 = QLabel('Script 2 Time: 00:00:00', self)

        # Create buttons
        self.run_script1_button = QPushButton('Run Script 1', self)
        self.run_script1_button.clicked.connect(self.run_script1)
        self.run_script1_button.setStyleSheet("background-color: lightblue; font-size: 16pt;")

        self.run_script2_button = QPushButton('Run Script 2', self)
        self.run_script2_button.clicked.connect(self.run_script2)
        self.run_script2_button.setStyleSheet("background-color: lightgreen; font-size: 16pt;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.timeLabel1)
        layout.addWidget(self.timeLabel2)
        layout.addWidget(self.run_script1_button)
        layout.addWidget(self.run_script2_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set a timer to update the time labels every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_times)
        self.timer.start(1000)  # Trigger every 1000 milliseconds (1 second)

    def update_times(self):
        if self.script1_runner.isRunning():
            self.script1_runner.timeUpdated.emit('script1', self.script1_runner.get_elapsed_time())
        if self.script2_runner.isRunning():
            self.script2_runner.timeUpdated.emit('script2', self.script2_runner.get_elapsed_time())

    def update_time1(self, script, elapsed_time):
        if script == 'script1':
            self.timeLabel1.setText(f'Script 1 Time: {elapsed_time}')

    def update_time2(self, script, elapsed_time):
        if script == 'script2':
            self.timeLabel2.setText(f'Script 2 Time: {elapsed_time}')

    def run_script1(self):
        if not self.script1_runner.isRunning():
            self.script1_runner.start()

    def run_script2(self):
        if not self.script2_runner.isRunning():
            self.script2_runner.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    script1_path = os.path.abspath(r"D:\MetaBCI\py_api\neuracle_py_api\example_online_read_data.py")
    script2_path = os.path.abspath(r"D:\MetaBCI\demos\brainstim_demos\stim_demo.py")