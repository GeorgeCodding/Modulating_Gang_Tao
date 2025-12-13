import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, resample, hilbert
from scipy.io.wavfile import write, read
import os

#-------------------------------------------------------------------
# STEP 1: Load and Preprocess
#-------------------------------------------------------------------

#open the file
filename = "gc.wav"
try:
    fs_source, x_raw = read(filename)
except FileNotFoundError:
    #simple A4 = 440 Hz sin wave if file not found
    fs_source = 44100
    t_dummy = np.linspace(0, 1, fs_source)
    x_raw = np.sin(2*np.pi*440*t_dummy) + 0.5*np.sin(2*np.pi*1000*t_dummy)

#convert from stereo to mono if stereo input
if x_raw.ndim > 1:
    x = x_raw.mean(axis=1)
else:
    x = x_raw

#normalize values to range of -1 to 1
if np.max(np.abs(x)) > 0:
    x = x.astype(float) / np.max(np.abs(x))
else:
    x = x.astype(float)

#-------------------------------------------------------------------
# STEP 1.5: Upsampling for FM Simulation
#-------------------------------------------------------------------

#FM Parameters
# - We use a high sample rate to capture the full bandwidth without aliasing
fc = 80000 #Carrier frequency is 80 kHz because this is typical for FM radio
fs_target = 600000 #600kHz oversamples the carrier by enough to ensure low distortion (this took some experimenting)

#resample the input to the target rate
num_samples_original = len(x)
duration_seconds = num_samples_original / fs_source
num_samples_new = int(duration_seconds * fs_target)

x_upsampled = resample(x, num_samples_new)
t = np.linspace(0, duration_seconds, len(x_upsampled), endpoint=False)
fs = fs_target

#-------------------------------------------------------------------
# STEP 2: Modulation
#-------------------------------------------------------------------

# IMPORTANT CHANGE: Increased kf from 2000 (from simple sin wave example) to 10k
# wider frequency deviation (higher kf) significantly improves audio quality by reducing background noise
kf = 10000  
integral_x = np.cumsum(x_upsampled) / fs
fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral_x)

#-------------------------------------------------------------------
# STEP 3: Demodulation with the Hilbert Transform
#-------------------------------------------------------------------

# IMPORTANT CHANGE: Switched from Slope Detection to Hilbert Transform
#derivative method (diff -> abs) is noisy and creates distortion -- Hilbert Transform extracts the instantaneous phase directly. 

#this is the analytic signal, meaning it has no negative frequency components
analytic_signal = hilbert(fm_signal)
#we then extract the instantaneous phase
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#we can find the frequency by taking the derivative of phase
instantaneous_freq = np.diff(instantaneous_phase, prepend=0) * fs / (2 * np.pi)
#then we remove the carrier to get the demodulated audio -- the signal is (fc + kf * x), so we subtract fc
demod_raw = instantaneous_freq - fc

#-------------------------------------------------------------------
# STEP 4: Filtering & Recovery
#-------------------------------------------------------------------

#IMPORTANT CHANGE: lowpass filter cutoff increased to 15kHz to capture all audio detail
lpf_cutoff = 15000 
sos_low = butter(10, lpf_cutoff, fs=fs, btype='low', output='sos')
demod_filtered = sosfiltfilt(sos_low, demod_raw)

#use a highpass filter to remove DC offset (20 Hz is lower than a person can hear)
hpf_cutoff = 20 
sos_high = butter(4, hpf_cutoff, fs=fs, btype='high', output='sos')
demod_clean = sosfiltfilt(sos_high, demod_filtered)

#-------------------------------------------------------------------
# STEP 5: Downsample & Export
#-------------------------------------------------------------------
#downsample back to the original rate to convert back to .wav

# 1. Create Output Folder
output_folder = "output_results_FM"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

demod_final = resample(demod_clean, num_samples_original)

demod_final = demod_final / kf

# 1. Define x_out (Clip the original signal 'x' to be safe for saving)
x_out = np.clip(x, -1.0, 1.0)
demod_final = np.clip(demod_final, -1.0, 1.0)

write(os.path.join(output_folder, "original_resampled_fm"), fs_source, np.int16(x_out * 32767))
write(os.path.join(output_folder, "demodulated_output_fm.wav"), fs_source, np.int16(demod_final * 32767))



#-------------------------------------------------------------------
# STEP 6: Visualization
#-------------------------------------------------------------------
#The How many points we output of the overall signal
start = 1000
end = 1000000


plt.figure(figsize=(14, 10))

# 1. Top Left: Input Time
plt.subplot(2, 2, 1)
plt.plot(t[start:end], x[start:end])
plt.title("Input Message (Time Domain)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.6)


# Top Right: Spectrogram Input
plt.subplot(2, 2, 2)
skip = int(fs_source * 0.01)
plt.specgram(x[skip:], Fs=fs_source, NFFT=1024, noverlap=512, cmap='inferno', vmin=-100)
plt.title(f"Original Input Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.ylim(0, 20000)


# 3. Bottom Left: Output Time
plt.subplot(2, 2, 3)
plt.plot(t[start:end], demod_final[start:end], color='green')
plt.title("FM Demodulated Output (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.6)

# Spectrogram Output
plt.subplot(2, 2, 4)
skip = int(fs_source * 0.01)
plt.specgram(demod_final[skip:], Fs=fs_source, NFFT=1024, noverlap=512, cmap='inferno', vmin=-100)
plt.title("FM Demodulated Output (Hilbert Method)")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.ylim(0, 20000)

plt.tight_layout()

save_path = os.path.join(output_folder, "Spectogram_Comparison.png")
plt.savefig(save_path)
plt.show()
