"""A simple rapid serial visual presentation (RSVP) speed reader.

Paste some text, choose a reading speed, and the words are flashed one at a
time in a large font. Reading can be paused/resumed with the Pause button or
the space bar, and two sliders tune the pace: the words-per-minute speed and a
multiplier that lengthens the pause on long words and at the end of sentences.

This version (speed_reader3) extends speed_reader2 by also showing the
estimated time left to finish the current text. The estimate counts down as
each word is shown, freezes while reading is paused, resets when new text is
read, and updates live when either slider is moved.
"""

from PyQt5 import QtWidgets, QtGui, QtCore


class SpeedReaderApp(QtWidgets.QWidget):
    """The main window: text input, the flashing word display and controls."""

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
        self.time_remaining_label = None
        self.text_input = None
        self.current_word_index = 0
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

        # Display area for the estimated time remaining
        self.time_remaining_label = QtWidgets.QLabel(self)
        self.time_remaining_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.time_remaining_label)
        self.update_time_remaining_label()

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
        """Split the input text into words and begin reading from the start."""
        self.words = self.text_input.toPlainText().split()
        self.current_word_index = 0
        self.display_next_word()

        # Reset the pause button
        self.paused = False
        self.pause_button.setText('Pause')
        self.timer.start()

    def display_next_word(self):
        """Show the next word and schedule the one after it (or stop at the end)."""
        if self.current_word_index < len(self.words):
            word = self.words[self.current_word_index]
            self.word_display.setText(word)
            delay = self.word_delay(word)

            self.timer.start(int(delay))
            self.current_word_index += 1
            self.update_time_remaining_label()
        else:
            self.word_display.setText('')  # Finished
            self.timer.stop()
            self.update_time_remaining_label()

    def word_delay(self, word):
        """The time (in milliseconds) that the given word is displayed for."""
        delay = 60000 / self.speed_slider.value()
        # Delay for long words
        if len(word) > 7:
            delay *= (self.long_word_slider.value() / 100)
        # Delay at end of sentences
        if "." in word:
            delay *= (self.long_word_slider.value() / 100)
        return delay

    def remaining_milliseconds(self):
        """Estimated time left to read the words that haven't been shown yet."""
        if not self.words:
            return 0
        return sum(self.word_delay(word)
                   for word in self.words[self.current_word_index:])

    def update_time_remaining_label(self):
        """Refresh the on-screen estimate, formatted as 'minutes:seconds'."""
        total_seconds = int(round(self.remaining_milliseconds() / 1000))
        minutes, seconds = divmod(total_seconds, 60)
        self.time_remaining_label.setText(
            'Time remaining: {:d}:{:02d}'.format(minutes, seconds))

    def toggle_pause(self):
        """Pause or resume reading (driven by the button and the space bar)."""
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
        # Speed affects how long the remaining text will take
        self.update_time_remaining_label()

    def update_long_word_label(self):
        self.long_word_label.setText(str(self.long_word_slider.value()))
        # The long-word multiplier affects how long the remaining text will take
        self.update_time_remaining_label()


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
