# Real-Time Audio Spectrum Visualizer

A Python application that creates a real-time 3D visualization of audio input using a contour plot. The visualizer displays frequency spectrum over time with amplitude represented in decibels (dB).

## Features

- Real-time 3D visualization of audio spectrum
- Adaptive noise floor reduction
- Frequency range from 20 Hz to 20 kHz
- Linear frequency binning for better visualization
- Dynamic dB scale adjustment
- Window function application for improved frequency analysis
- Support for custom audio input device selection

## Requirements

```
sounddevice
numpy
matplotlib
scipy
```

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install sounddevice numpy matplotlib scipy
```

## Usage

Run the script using Python:

```bash
python main.py
```

When launched, the program will:
1. Display a list of available audio input devices
2. Prompt you to select an input device (press Enter for default device)
3. Open a 3D visualization window showing the real-time audio spectrum

## Technical Details

### Visualization Parameters

- Sample Rate: 44.1 kHz
- Chunk Size: 1024 samples
- History Length: 60 seconds
- Frequency Range: 20 Hz - 20 kHz
- Number of Frequency Bins: 200
- Update Interval: 50ms

### Signal Processing

The visualizer implements several signal processing techniques:

1. **Windowing**: Applies a Hann window to reduce spectral leakage
2. **FFT Analysis**: Computes the frequency spectrum using Fast Fourier Transform
3. **Linear Frequency Mapping**: Remaps FFT bins to linear frequency scale
4. **Noise Floor Adaptation**: Dynamically adjusts noise floor for cleaner visualization
5. **dB Conversion**: Converts amplitude to decibels for better dynamic range representation

### Visualization Features

- **3D Contour Plot**: Shows frequency content over time
- **Dynamic Scaling**: Automatically adjusts amplitude scale
- **Color Mapping**: Uses 'coolwarm' colormap for amplitude visualization
- **Labeled Axes**: Clear frequency and time labels
- **Customizable View**: 3D perspective can be rotated and zoomed

## Class Structure

The main `DirectReplacementContour3DVisualizer` class handles:
- Audio input stream management
- Real-time signal processing
- 3D visualization updates
- Buffer management for historical data

## Key Methods

- `setup_plot()`: Initializes the 3D visualization environment
- `process_audio()`: Processes incoming audio data
- `amplitude_to_db()`: Converts linear amplitude to decibels
- `update_plot()`: Updates the visualization
- `start_stream()`: Manages the audio input stream

## Customization

You can modify several parameters in the code to adjust the visualization:
- `CHUNK`: Buffer size for audio processing
- `RATE`: Sample rate
- `SECONDS`: Duration of history to display
- `MIN_FREQ`/`MAX_FREQ`: Frequency range
- `num_bins`: Number of frequency bins
- `noise_adaptation_rate`: Speed of noise floor adaptation

## Troubleshooting

If you encounter issues:

1. **No Audio Input**:
   - Check if your input device is properly selected
   - Verify microphone permissions
   - Test with different input devices

2. **Performance Issues**:
   - Reduce the history length (`SECONDS`)
   - Decrease the number of frequency bins (`num_bins`)
   - Increase the update interval in `FuncAnimation`

3. **Display Problems**:
   - Ensure matplotlib backend is compatible with your system
   - Update graphics drivers
   - Try different matplotlib backends

## License

This project is open source. Feel free to use, modify, and distribute as needed.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.