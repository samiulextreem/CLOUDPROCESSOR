import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from numpy.fft import fft, ifft, fftfreq

# ===============================
# Generate Synthetic Signal
# ===============================
np.random.seed(0)
n = 500
t = np.linspace(0, 1, n)
original_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

# Add Gaussian noise
noise = np.random.normal(0, 0.5, n)
noisy_signal = original_signal + noise

# ===============================
# 1. Adaptive Wiener Filter (Local)
# ===============================
adaptive_filtered = wiener(noisy_signal)

# ===============================
# 2. Frequency-Domain Wiener Filter
# ===============================
# FFT of signals
Y = fft(noisy_signal)
S = fft(original_signal)
N = fft(noise)

# Power Spectral Densities
S_xx = np.abs(S) ** 2
S_nn = np.abs(N) ** 2

# Wiener filter transfer function
H = S_xx / (S_xx + S_nn + 1e-10)  # small term avoids divide-by-zero

# Apply Wiener filter in frequency domain
X_hat = H * Y
frequency_filtered = np.real(ifft(X_hat))

# ===============================
# Plot Comparison
# ===============================
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal, label="Noisy Signal", color="gray")
plt.plot(t, original_signal, label="Original Signal", linestyle="--", color="green")
plt.title("Noisy vs Original Signal")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, adaptive_filtered, label="Adaptive Wiener Filter", color="blue")
plt.plot(t, original_signal, label="Original Signal", linestyle="--", color="green")
plt.title("Adaptive (Local) Wiener Filter")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, frequency_filtered, label="Frequency-Domain Wiener Filter", color="red")
plt.plot(t, original_signal, label="Original Signal", linestyle="--", color="green")
plt.title("Frequency-Domain Wiener Filter")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
