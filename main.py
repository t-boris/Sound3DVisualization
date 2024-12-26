import sounddevice as sd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

matplotlib.use('TkAgg')


class DirectReplacementContour3DVisualizer:
    def __init__(self, device=None):
        self.contour = None
        self.Y = None
        self.X = None
        self.ax = None
        self.fig = None
        self.CHUNK = 1024
        self.RATE = 44100
        self.SECONDS = 60
        self.device = device
        self.chunks_per_second = self.RATE / self.CHUNK
        self.HISTORY = int(self.chunks_per_second * self.SECONDS)
        self.MIN_FREQ = 20
        self.MAX_FREQ = 20000
        self.freq_bins = np.fft.rfftfreq(self.CHUNK, 1 / self.RATE)
        self.freq_mask = (self.freq_bins >= self.MIN_FREQ) & (self.freq_bins <= self.MAX_FREQ)
        self.filtered_bins = self.freq_bins[self.freq_mask]
        self.num_bins = 200  # Linear frequency bins
        self.linear_freq_points = np.linspace(self.MIN_FREQ, self.MAX_FREQ, num=self.num_bins)
        self.buffer = np.zeros((self.HISTORY, self.num_bins - 1))
        self.window = signal.windows.hann(self.CHUNK)
        self.noise_floor = np.ones(self.num_bins - 1) * 1e-6
        self.noise_adaptation_rate = 0.01
        self.reference_amplitude = 1e-5  # Reference amplitude for dB calculation
        self.max_db = 60  # Initialize max dB for dynamic scaling
        self.min_db = 0  # Minimum dB level to display
        self.setup_plot()

    def setup_plot(self):
        """
        Sets up the 3D contour plot for visualizing audio data with dB scale.
        """
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        time_points = np.linspace(0, self.SECONDS, self.HISTORY)
        freq_points = np.linspace(0, 1, self.num_bins - 1)
        self.X, self.Y = np.meshgrid(freq_points, time_points)
        self.contour = None
        self.ax.set_xlabel('Frequency (Hz)', labelpad=10)
        self.ax.set_ylabel('Time (s)', labelpad=10)
        self.ax.set_zlabel('Amplitude (dB)', labelpad=10)  # Updated label
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(self.SECONDS, 0)
        self.ax.set_zlim(self.min_db, self.max_db)  # Initial zlim in positive dB

        # Set up frequency axis ticks
        freq_ticks = np.linspace(self.MIN_FREQ, self.MAX_FREQ, 10)
        tick_positions = (freq_ticks - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels([f"{int(freq)}" for freq in freq_ticks])

        # Set up time axis ticks
        time_ticks = np.arange(0, self.SECONDS + 1, 10)
        self.ax.set_yticks(time_ticks)
        self.ax.set_yticklabels([f"{t}s" for t in reversed(time_ticks)])

        self.ax.view_init(elev=30, azim=230)
        self.ax.set_box_aspect([2, 1.5, 1])

    def amplitude_to_db(self, amplitude):
        """
        Convert amplitude values to positive decibels relative to reference amplitude.
        """
        # Add small epsilon to avoid log of zero
        epsilon = 1e-10
        return 20 * np.log10((amplitude + epsilon) / self.reference_amplitude)

    def process_audio(self, audio_data):
        # Apply window function
        windowed_data = audio_data * self.window

        # Compute FFT and get magnitude spectrum
        fft = np.abs(np.fft.rfft(windowed_data))
        fft = fft[self.freq_mask]

        # Map FFT to linear frequency bins
        linear_fft = np.zeros(self.num_bins - 1)
        for i in range(self.num_bins - 1):
            mask = (self.filtered_bins >= self.linear_freq_points[i]) & (
                    self.filtered_bins < self.linear_freq_points[i + 1])
            if np.any(mask):
                linear_fft[i] = np.mean(fft[mask])

        # Update adaptive noise floor
        self.noise_floor = (1 - self.noise_adaptation_rate) * self.noise_floor + \
                           self.noise_adaptation_rate * np.maximum(linear_fft, self.noise_floor)

        # Apply noise threshold and convert to dB
        linear_fft = np.maximum(linear_fft - self.noise_floor, 0)
        db_values = self.amplitude_to_db(linear_fft)

        # Clip values to minimum dB threshold
        db_values = np.maximum(db_values, self.min_db)

        # Update buffer with dB values
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1, :] = db_values

        # Update maximum dB value for scaling
        self.max_db = max(self.max_db, np.max(db_values))

    def update_plot(self, frame):
        # Adjust z-axis limits based on the dB range
        self.ax.set_zlim(self.min_db, max(60, int(self.max_db * 1.1)))

        # Remove old contour plot
        if self.contour:
            self.contour.remove()

        # Create new contour plot with updated dB values
        self.contour = self.ax.contour3D(
            self.X, self.Y, self.buffer,
            levels=np.linspace(self.min_db, max(0, self.max_db), 50),
            cmap='coolwarm',
            alpha=0.9
        )
        return self.contour,

    def audio_callback(self, indata, frames, time, status):
        self.process_audio(indata[:, 0])

    def start_stream(self):
        stream = sd.InputStream(
            callback=self.audio_callback,
            device=self.device,
            channels=1,
            samplerate=self.RATE,
            blocksize=self.CHUNK
        )
        with stream:
            ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)
            plt.show()


if __name__ == "__main__":
    print("Available audio devices:")
    print(sd.query_devices())
    device_id = input("Select input device ID (leave blank for default): ")
    device_id = int(device_id) if device_id else None
    visualizer = DirectReplacementContour3DVisualizer(device=device_id)
    visualizer.start_stream()