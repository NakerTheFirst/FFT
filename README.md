# Fourier Transforms
A Python implementation of 1D and 2D Fourier Transform techniques including Discrete Fourier Transform (DFT), Fast Fourier Transform (FFT), Inverse Discrete Fourier Transform (IDFT), and Inverse Fast Fourier Transform (IFFT). These techniques are fundamental in the field of signal processing for analysing the frequency components of signals and images.

<p align="center"><img width="659" src="https://github.com/NakerTheFirst/FFT/blob/main/2d_signal.png" alt="Image of a table of coefficient values for LSM-based linear regression"></p>
<p align="center"><a href="report.pdf">Figure 1.<alt="Report link"></a> Transformed 2D signal in spatial domain</p>

A full, detailed report of the process in Polish can be read in 
<a href="report.pdf">report.pdf</a>

## Features
- **DFT and FFT**: Compute the frequency spectrum of one-dimensional signals.
- **IDFT and IFFT**: Reconstruct signals from their frequency components.
- **2D FFT**: Extend the analysis to two-dimensional signals or images.
- **Amplitude and Noise Analysis**: Analyse the amplitude responses and noise characteristics of signals.
- **Visualisations**: Visualise signals, frequency spectra, and noise in various graphical formats.
  
## Prerequisites
To run the scripts in this repository, you will need Python along with several libraries. Ensure you have the following installed:
- Python 3.6+
- NumPy
- Matplotlib

These can be installed using pip if not already installed:

```bash
pip install numpy matplotlib
```

## Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/FFT.git
```

## Usage
Navigate to the project directory and execute the main script to start the analysis:

```bash
cd FFT
python main.py
```

The script will prompt for input regarding the dimensionality of the data (1D or 2D) and process accordingly. The results including graphical outputs will be displayed during the run.
