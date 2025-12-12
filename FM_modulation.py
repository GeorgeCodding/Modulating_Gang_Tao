import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write

#-------------------------------------------------------------------
# STEP 1: Set up the parameters
#-------------------------------------------------------------------
fs = 45000
duration = 1.0
f_in = 600 #600 Hz input message
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
x = np.sin(2*np.pi*f_in*t)

plt.figure(figsize=(10,4))
plt.plot(t[:1000], x[:1000])
plt.title("600 Hz Input Message Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
plt.close()

#-------------------------------------------------------------------
# STEP 2: FM Modulation
#-------------------------------------------------------------------
fc = 5000  #5 kHz Carrier
kf = 2000  #frequency deviation sensitivity (controls how much the freq shifts from the baseline freq)

#our modulated signal is y(t) = A * cos(2*pi*fc*t + 2*pi*kf * integral(x))
#np.cumsum approximates the integral of the message signal
integral_x = np.cumsum(x) / fs

fm_signal = np.cos( ((2*np.pi)*fc*t) + ((2*np.pi)*kf*integral_x) )

plt.figure(figsize=(10,4))
plt.plot(t[:1000], fm_signal[:1000])
plt.title("FM Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
plt.close()

#-------------------------------------------------------------------
# STEP 3: FM Demodulation (Slope Detection)
#-------------------------------------------------------------------

#we can't demodulate FM using envelope detector because the amplitude is constant.
#so first we have to differentiate
#use np.diff to take the derivative (np.diff reduces array size by 1, so we append a 0 to match length)
slope_detected = np.diff(fm_signal, prepend=0)

#the derivative looks like d(y(t))/dt = -A*( 2*pi*fc + 2*pi*kf*x(t) ) * sin(2*pi*fc*t + 2*pi*kf*integral(x))

#now, the frequency of the modulated signal is part of the amplitude
#we've converted frequency variation to amplitude variation, so we can use an envelope detector!!! yay!!!

#envelope detector is used just like with AM
envelope = np.abs(slope_detected)

#-------------------------------------------------------------------
# STEP 4: Lowpass filtering
#-------------------------------------------------------------------

#(Same as before: cutoff must be > [however many Hz, in this case 600] Hz message, and < Carrier)
if (f_in*2 < .8*fc):
    lpf_cutoff = f_in*1.5
else:
    lpf_cutoff = (fc-f_in)/2

lpf_filter_order = 8 #I like the number 8
b, a = butter(lpf_filter_order, lpf_cutoff / (fs/2), btype='low')
demod_fm = filtfilt(b, a, envelope)
demod_fm = demod_fm - np.mean(demod_fm)
#remove DC offset

#-------------------------------------------------------------------
# STEP 4.5: DC offset removal and amplitude matching
#-------------------------------------------------------------------

#To figure out the scalar multiple we need to make the amplitude of the demodulated signal match the original, we need to know the total gain
#that we've accumulated through the transmission process.

#FIRST, the additional gain from taking the derivative during step 3 is (2*pi*kf)

#SECOND, since np.diff is discrete, np.diff ~= (dy/dx)*delta_t
#another way to write this is np.diff ~= (dy/dx)*(1/fs)
#now, the toal gain we've accumulated is (2*pi*kf/fs)

#THIRD, when we feed our modulated signal through an envelope detector and filter
#we multiply by the average of the rectified sinusoid (2/pi)
#now, the total gain we've accumulated is (2*pi*kf/fs * 2/pi) = 4*kf/fs

#so to undo this gain and match the demodulated amplitude to that of the original signal, we multiply by fs/(4*kf)
demod_fm = (fs/(4*kf))*demod_fm

plt.figure(figsize=(10,4))
#skip the first 200 samples to hide the spike in the beginning as the lowpass filter settles
start_index = 200
plt.plot(t[start_index:2000], demod_fm[start_index:2000])
plt.title("Demodulated FM Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
plt.close()

#-------------------------------------------------------------------
# STEP 5: Convert original signal and demodulated signal to .wav files for comparison
#-------------------------------------------------------------------

#ensure values are strictly between -1.0 and 1.0 (WAV files can glitch if float data exceeds this range)
#for both input and output
x = np.clip(x, -1.0, 1.0)
demod_fm = np.clip(demod_fm, -1.0, 1.0)

#convert to 16-bit PCM format (Standard for WAV) -- this maps the float range -1.0...1.0 to the integer range -32768...32767
#for both input and output
scaled_audio_input = np.int16(x * 32767)
write("original_signal_fm.wav", fs, scaled_audio_input)
scaled_audio_output = np.int16(demod_fm * 32767)
write("demodulated_fm.wav", fs, scaled_audio_output)

#-------------------------------------------------------------------
# STEP 6: Spectrogram Comparison
#-------------------------------------------------------------------
plt.figure(figsize=(12, 10))

#spectrogram of original input
plt.subplot(2, 1, 1)
#NFFT determines frequency resolution -- higher = finer frequency lines.
plt.specgram(x, Fs=fs, NFFT=2048, noverlap=1024, cmap='inferno')
plt.title("Spectrogram of Original Input (600 Hz Pure Tone)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
plt.ylim(0, 2000) #zoom in to 0-2000 Hz since our signal is at 600 Hz

#spectrogram of demodulated output
plt.subplot(2, 1, 2)
#just like when displaying the graphs, we ignore the beginning of the demodulated signal as the lowpass filter settles

#DEFUALT PLOTTING
# plt.specgram(demod_fm[200:], Fs=fs, NFFT=2048, noverlap=1024, cmap='inferno')

#OPTIONAL: uncomment to hide quiet noise at -60 to -40 dB
plt.specgram(demod_fm[200:], Fs=fs, NFFT=2048, noverlap=1024, cmap='inferno', vmin=-60)

plt.title("Spectrogram of Demodulated Output")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
plt.ylim(0, 2000)

plt.tight_layout()
plt.show()
plt.close()
