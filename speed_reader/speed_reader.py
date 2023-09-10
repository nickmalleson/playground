import tkinter as tk
from tkinter import ttk
import time

class EnhancedSpeedReader(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Speed Reader')
        self.geometry('600x400')

        # Text input area
        self.text_input = tk.Text(self, wrap='word')
        self.text_input.pack(pady=20, padx=20)

        # Display area for words
        self.word_display = tk.Label(self, text='', font=('Arial', 24))
        self.word_display.pack(pady=20)

        # Sliders for configuration
        self.speed_slider = tk.Scale(self, from_=100, to=1000, orient='horizontal', label='Words per minute')
        self.speed_slider.set(300)
        self.speed_slider.pack(pady=10)

        self.long_word_slider = tk.Scale(self, from_=1, to=3, orient='horizontal', label='Multiplier for long words', resolution=0.1)
        self.long_word_slider.set(1.5)
        self.long_word_slider.pack(pady=10)

        # Start button
        self.start_button = ttk.Button(self, text='Start Reading', command=self.start_reading)
        self.start_button.pack(pady=10)

        # Pause/Resume button
        self.paused = False
        self.pause_button = ttk.Button(self, text='Pause', command=self.toggle_pause)
        self.pause_button.pack(pady=10)

    def start_reading(self):
        text = self.text_input.get('1.0', 'end-1c').split()
        for word in text:
            while self.paused:
                self.update()
                time.sleep(0.1)
            self.word_display.config(text=word)
            delay = 60.0 / self.speed_slider.get()
            if len(word) > 7:  # considering words with more than 7 characters as long words
                delay *= self.long_word_slider.get()
            time.sleep(delay)
            self.update()
        self.word_display.config(text='Finished!')

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text='Resume')
        else:
            self.pause_button.config(text='Pause')

# Uncomment the lines below to run the GUI on your local machine
app = EnhancedSpeedReader()
app.mainloop()

