from PyQt5 import QtWidgets, QtGui, QtCore


class SpeedReaderApp(QtWidgets.QWidget):
    DISPLAY_FONT_SIZE = 92
    INITIAL_LONG_WORD_DELAY = 150
    INITIAL_READ_SPEED = 490
    INITIAL_WINDOW_WIDTH = 800

    def __init__(self):
        super().__init__()
        self.paused = None
        self.words = None
        self.timer = None
        self.pause_button = None
        self.start_button = None
        self.long_word_label = None
        self.long_word_slider = None
        self.speed_label = None
        self.speed_slider = None
        self.word_display = None
        self.text_input = None
        self.init_ui()

    def init_ui(self):
        # Layouts
        layout = QtWidgets.QVBoxLayout()
        controls_layout = QtWidgets.QHBoxLayout()

        # Text input area
        self.text_input = CustomTextEdit(self)
        self.text_input.space_pressed.connect(self.toggle_pause)
        self.text_input.setMinimumWidth(SpeedReaderApp.INITIAL_WINDOW_WIDTH)
        layout.addWidget(self.text_input)

        # Display area for words
        self.word_display = QtWidgets.QLabel(self)
        self.word_display.setFont(QtGui.QFont('Arial', SpeedReaderApp.DISPLAY_FONT_SIZE))
        self.word_display.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.word_display)

        # Sliders for configuration
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.speed_slider.setMinimum(300)
        self.speed_slider.setMaximum(700)
        self.speed_slider.setValue(SpeedReaderApp.INITIAL_READ_SPEED)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        controls_layout.addWidget(QtWidgets.QLabel('Words per minute'))
        controls_layout.addWidget(self.speed_slider)
        self.speed_label = QtWidgets.QLabel(str(self.speed_slider.value()))
        controls_layout.addWidget(self.speed_label)

        self.long_word_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.long_word_slider.setMinimum(100)
        self.long_word_slider.setMaximum(200)
        self.long_word_slider.setValue(SpeedReaderApp.INITIAL_LONG_WORD_DELAY)
        self.long_word_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.long_word_slider.setTickInterval(10)
        self.long_word_slider.valueChanged.connect(self.update_long_word_label)
        controls_layout.addWidget(QtWidgets.QLabel('Multiplier for long words'))
        controls_layout.addWidget(self.long_word_slider)
        self.long_word_label = QtWidgets.QLabel(str(self.long_word_slider.value()))
        controls_layout.addWidget(self.long_word_label)

        layout.addLayout(controls_layout)

        # Start button
        self.start_button = QtWidgets.QPushButton('Start Reading', self)
        self.start_button.clicked.connect(self.start_reading)
        layout.addWidget(self.start_button)

        # Pause/Resume button
        self.pause_button = QtWidgets.QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pause_button)

        # Timer for reading
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.display_next_word)
        self.words = []
        self.paused = False

        self.setLayout(layout)
        self.setWindowTitle('Speed Reader')
        self.show()


    def start_reading(self):
        self.words = self.text_input.toPlainText().split()
        self.current_word_index = 0
        self.display_next_word()

        # Reset the pause button
        self.paused = False
        self.pause_button.setText('Pause')
        self.timer.start()

    def display_next_word(self):
        if self.current_word_index < len(self.words):
            word = self.words[self.current_word_index]
            self.word_display.setText(word)
            delay = 60000 / self.speed_slider.value()
            # Delay for long words
            if len(word) > 7:
                delay *= (self.long_word_slider.value() / 100)
            # Delay at end of sentences
            if "." in word:
                delay *= (self.long_word_slider.value() / 100)

            self.timer.start(int(delay))
            self.current_word_index += 1
        else:
            self.word_display.setText('')  # Finished
            self.timer.stop()

    def toggle_pause(self):
        if self.paused:
            self.paused = False
            self.pause_button.setText('Pause')
            self.timer.start()
        else:
            self.paused = True
            self.pause_button.setText('Resume')
            self.timer.stop()

    def update_speed_label(self):
        self.speed_label.setText(str(self.speed_slider.value()))

    def update_long_word_label(self):
        self.long_word_label.setText(str(self.long_word_slider.value()))


# This is to get the space bar to pause/resume the reading
# (otherwise the text input area would capture the space key)
class CustomTextEdit(QtWidgets.QTextEdit):
    space_pressed = QtCore.pyqtSignal()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Space:
            self.space_pressed.emit()
        else:
            super().keyPressEvent(event)

# Uncomment the lines below to run the GUI on your local machine
app = QtWidgets.QApplication([])
window = SpeedReaderApp()
app.exec_()
