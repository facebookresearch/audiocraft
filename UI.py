import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QTextEdit, QLabel, QFormLayout, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess

class TrainingThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, command):
        QThread.__init__(self)
        self.command = command

    def run(self):
        process = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            self.update_signal.emit(line)
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            self.update_signal.emit(f"Process finished with return code {return_code}")

class AudiocraftGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audiocraft - Audio Processing Tool")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()
        self.team_input = QLineEdit()
        self.cluster_input = QLineEdit()
        env_form_layout = QFormLayout()
        env_form_layout.addRow('Team:', self.team_input)
        env_form_layout.addRow('Cluster:', self.cluster_input)
        self.env_submit_button = QPushButton('Set Environment')
        self.env_submit_button.clicked.connect(self.set_environment)
        env_form_layout.addRow(self.env_submit_button)

        self.config_input = QTextEdit()
        self.config_input.setPlaceholderText("Enter the training command, e.g., 'python train.py --config config.yaml'")
        self.training_log_display = QTextEdit()
        self.training_log_display.setReadOnly(True)
        self.train_submit_button = QPushButton('Start Training')
        self.train_submit_button.clicked.connect(self.start_training)

        main_layout.addLayout(env_form_layout)
        main_layout.addWidget(QLabel("Training Command:"))
        main_layout.addWidget(self.config_input)
        main_layout.addWidget(self.train_submit_button)
        main_layout.addWidget(QLabel("Training Log:"))
        main_layout.addWidget(self.training_log_display)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_environment(self):
        team = self.team_input.text()
        cluster = self.cluster_input.text()
        os.environ['AUDIOCRAFT_TEAM'] = team
        if cluster:
            os.environ['AUDIOCRAFT_CLUSTER'] = cluster
        self.update_log(f"Environment set: AUDIOCRAFT_TEAM={team}, AUDIOCRAFT_CLUSTER={cluster}")

    def start_training(self):
        command = self.config_input.toPlainText()
        if not command.strip():
            QMessageBox.warning(self, "Input Error", "Please enter a valid command to run.")
            return
        self.train_submit_button.setDisabled(True)
        self.training_thread = TrainingThread(command)
        self.training_thread.update_signal.connect(self.update_log)
        self.training_thread.start()
        self.training_thread.finished.connect(lambda: self.train_submit_button.setDisabled(False))

    def update_log(self, text):
        self.training_log_display.append(text)
        self.training_log_display.verticalScrollBar().setValue(self.training_log_display.verticalScrollBar().maximum())

def main():
    app = QApplication(sys.argv)
    main_window = AudiocraftGUI()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
