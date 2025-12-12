# THIS IS THE AM CODE — UPDATED WITH SPECTROGRAM COMPARISON AND .WAV FILE SAVING
import matplotlib
matplotlib.use('TkAgg')  # Forces standard pop-up windows

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
from scipy.io.wavfile import read, write
import os
#-------------------------------------------------------------------
# STEP 1: Load WAV File
#-------------------------------------------------------------------

# 1. Load the file
filename = "gc.wav"

fs, x_raw = read(filename)
fs_orig = fs
# 2. Handle Stereo (if file has 2 channels, mix to mono)
# The AM math (1 + x) expects a 1D array.
if x_raw.ndim > 1:
    x = x_raw.mean(axis=1)  # Average left and right channels
else:
    x = x_raw

# 3. Normalize to float (-1.0 to 1.0)
# WAV files are typically integers. We need floats for modulation math.
if np.max(np.abs(x)) > 0:
    x = x.astype(float) / np.max(np.abs(x))
else:
    x = x.astype(float)

# 4. Create Time Array (t) to match the new file length
duration = len(x) / fs
t = np.linspace(0, duration, len(x), endpoint=False)

# 5. Define Max Audio Frequency (Critical for Filter Logic)
# Your old code used 'f_in' (600) to calculate the filter cutoff later.
# Since we now have complex audio, we estimate the max voice/music freq.
f_in = 5000  # Estimate: 3kHz is typical for AM radio voice bandwidth

#-------------------------------------------------------------------
# STEP 1: Set up the parameters
#-------------------------------------------------------------------

# 1. Define target AM parameters
fc = 80000        # Carrier Frequency (40 kHz)
fs_target = 600000 # Target System Sample Rate (160 kHz)

# 2. Calculate the Ratio
# We know the file's original 'fs' (from read(filename))
# We need to stretch the data to fit the new 'fs_target'
num_samples_original = len(x)
duration_seconds = num_samples_original / fs
num_samples_new = int(duration_seconds * fs_target)

# 3. Perform Resampling
# This interpolates the audio to create the extra data points needed to match the Am transmitter running at 160,000 samples per second
x = resample(x, num_samples_new)

# 4. Update the global variables for the rest of the script
fs = fs_target
t = np.linspace(0, duration_seconds, len(x), endpoint=False)

#-------------------------------------------------------------------
# STEP 2: AM Modulation
#-------------------------------------------------------------------

#AM Equation: y(t) = (A + x(t)) * cos(2*pi*fc*t)
carrier_wave = np.cos(2 * np.pi * fc * t)
#amplitude of 1 for this demo
am_signal = (1 + x) * carrier_wave



#-------------------------------------------------------------------
# STEP 4: Lowpass filtering
#-------------------------------------------------------------------

#final am step: demodulation using envelope detector circuit (diode + low pass filter combo)
    #source: https://digilent.com/blog/pushing-the-envelope-detector-exploring-demodulation-am-and-fm-signals/
envelope = np.abs(am_signal)

#sideband calcultions:
    #upper sideband: 5000+600 = 5600 Hz
    #lower sideband: 5000-600 = 4400 Hz
    #carrier: 5000 Hz

#lowpass cutoff frequency must be > message (600 Hz, in this case) but < carrier (5000 Hz)
#this if-else block tries to keep the cutoff frequency as low as possible so as to eliminate noisy harmonics
if (f_in*2 < .8*fc):
    lpf_cutoff = f_in*1.5
else:
    lpf_cutoff = (fc-f_in)/2
lpf_filter_order = 8 #8!

#create the butterworth filter!
#normalized cutoff = cutoff/nyquist
#fs/2: nyquist sampling
b, a = butter(lpf_filter_order, lpf_cutoff / (fs/2), btype='low')
demod_am = filtfilt(b, a, envelope)

#-------------------------------------------------------------------
# STEP 4.5: DC Removal and Amplitude Matching
#-------------------------------------------------------------------

# 1. APPLY HIGH-PASS FILTER (Simulates a Series Capacitor / AC Coupling)
# We want to block DC (0 Hz) but keep our audio (600 Hz).
# A cutoff of 20 Hz is standard for audio to remove DC without hurting bass.
hpf_cutoff = 20
hpf_order = 4 # Steep enough to kill DC

# Note: btype='high' creates the high-pass filter
b_hp, a_hp = butter(hpf_order, hpf_cutoff / (fs/2), btype='high')

# Apply the filter to the Low-Passed signal from Step 4
demod_am = filtfilt(b_hp, a_hp, demod_am)
#SECOND, scale the amplitude
# - when we rectify a cosine wave (take abs) and filter it, we get the AVERAGE value.
# - the average value of a fully rectified cosine wave is 2/pi (~0.637) of the peak, so we multiply by pi/2
scaling_factor = np.pi / 2
demod_am = demod_am * scaling_factor

#-------------------------------------------------------------------
# STEP 5: Export to "output_results" Folder
#-------------------------------------------------------------------

output_folder = "output_results_AM"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. Save WAV Files
# Clip to safe audio range
x_out = np.clip(x, -1.0, 1.0)
demod_am = np.clip(demod_am, -1.0, 1.0)

# Downsample back to original source rate (e.g. 44100 Hz)
demod_am = resample(demod_am, num_samples_original) 
x_out = resample(x, num_samples_original)

write(os.path.join(output_folder, "original_resampled.wav"), int(fs_orig), np.int16(x_out * 32767))
write(os.path.join(output_folder, "demodulated_output_AM.wav"), int(fs_orig), np.int16(demod_am * 32767))

#-------------------------------------------------------------------
# FINAL PLOTTING: Save and Display
#-------------------------------------------------------------------

# Slice for plotting (viewing a segment of the wave)
# We look at 2000 samples to see the carrier waves clearly
start = 10000
end = 1000000


# --- IMAGE 2: The Grid Summary ---
plt.figure(figsize=(14, 10))

# 1. Top Left: Input Time
plt.subplot(2, 2, 1)
plt.plot(t[start:end], x_out[start:end])
plt.title("Input Message (Time Domain)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.6)

# 2. Top Right: Input Spectrogram
plt.subplot(2, 2, 2)

skip = int(fs * 0.01)
plt.specgram(x_out[skip:], Fs= fs_orig, NFFT=1024, noverlap=512, cmap='inferno', vmin=-100)

plt.title("Spectrogram of Input")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 20000) # <--- FIX 2: See the full 20kHz range


# 3. Bottom Left: Output Time
plt.subplot(2, 2, 3)
plt.plot(t[start:end], demod_am[start:end], color='green')
plt.title("Demodulated Output (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.6)

# 4. Bottom Right: Output Spectrogram
plt.subplot(2, 2, 4)
plt.specgram(demod_am[skip:], Fs=fs_orig, NFFT=1024, noverlap=512, cmap='inferno', vmin=-100)
plt.title("Spectrogram of Output")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 20000)

plt.tight_layout()

# Save and Show
save_path2 = os.path.join(output_folder, "2_Grid_Summary.png")
plt.savefig(save_path2)
plt.show()