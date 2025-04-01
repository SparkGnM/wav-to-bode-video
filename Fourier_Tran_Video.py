import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import cv2
import librosa
import librosa.display
import os
from moviepy import VideoFileClip, AudioFileClip

# ===========================
# 1. Load and Save Audio File
# ===========================
file_name = "Frustration.wav"  # Update with your file name
output_dir = os.getcwd()  # Update path

# Create directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Created directory: {output_dir}")

# Load audio file
audio_data, Fs = librosa.load(file_name, sr=None)  # Preserve original sampling rate

# Save the processed audio (optional)
output_audio_path = os.path.join(output_dir, "processed_audio.wav")
wav.write(output_audio_path, Fs, (audio_data * 32767).astype(np.int16))
print(f"🎧 Audio saved to: {output_audio_path}")

# ==========================
# 2. Setup FFT Parameters
# ==========================
window_size = 8192  # Match the window size to nfft
overlap = 2048  # 50% overlap for smoother transitions
nfft = 8192  # High-resolution FFT

# Define window for FFT (Hamming window to reduce spectral leakage)
window_function = np.hamming(window_size)

# Set frequency range for visualization
min_freq = 20  # Minimum frequency in Hz
max_freq = 20000  # Maximum frequency in Hz

# =======================
# 3. Generate Spectrogram
# =======================
D = librosa.stft(audio_data, n_fft=nfft, hop_length=overlap, window=window_function)
S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

freq_axis = librosa.fft_frequencies(sr=Fs, n_fft=nfft)
time_axis = librosa.times_like(D, sr=Fs, hop_length=overlap)

# Limit frequency range to 20 Hz - 20 kHz
freq_mask = (freq_axis >= min_freq) & (freq_axis <= max_freq)
S_dB_limited = S_dB[freq_mask, :]
freq_axis_limited = freq_axis[freq_mask]

# Plot and save initial spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_dB_limited, sr=Fs, hop_length=overlap, x_axis="time", y_axis="log", cmap="viridis", fmin=min_freq, fmax=max_freq)
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Audio (20 Hz to 20 kHz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
spectrogram_path = os.path.join(output_dir, "spectrogram_plot.png")
plt.savefig(spectrogram_path)
plt.close()
print(f"📊 Spectrogram saved to: {spectrogram_path}")

# =============================
# 4. Create Video with FFT Frames
# =============================
video_name = os.path.join(output_dir, "fourier_visualization.mp4")  # Change extension to .mp4
frame_rate = 30  # FPS for video
frame_step = int(Fs / frame_rate)  # Time step between frames

# Use mp4v codec for video encoding (fallback if X264 isn't working)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to mp4v
height, width = 600, 1000
video_out = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Fixed amplitude scale
max_amplitude = 160  # Set the maximum magnitude scale

num_frames = len(audio_data) // frame_step
for i in range(num_frames):
    start = i * frame_step
    end = start + window_size
    if end > len(audio_data):
        break

    segment = audio_data[start:end] * window_function[: end - start]
    fft_data = np.fft.fft(segment, nfft)
    fft_magnitude = np.abs(fft_data[: nfft // 2])
    freq_axis_fft = np.fft.fftfreq(nfft, d=1 / Fs)[: nfft // 2]

    # Normalize the FFT magnitude to a fixed range
    fft_magnitude_normalized = np.clip(fft_magnitude, -max_amplitude, max_amplitude)

    # Plot FFT magnitude for 20 Hz to 20 kHz
    plt.figure(figsize=(10, 6))

    glow_alpha = 0.6  # Initial alpha for glow
    for line_width in range(1, 6):  # 5 layers for glow effect
        plt.plot(freq_axis_fft, fft_magnitude_normalized, color="hotpink", linewidth=line_width, alpha=glow_alpha)
        glow_alpha *= 0.5  # Reduce transparency for each layer

    # Customizations
    plt.plot(freq_axis_fft, fft_magnitude_normalized, color="hotpink")  # Line color (change this)
    plt.xlim(min_freq, max_freq)
    plt.ylim(-max_amplitude, max_amplitude)  # Lock y-axis amplitude scale

    # Create symmetry by plotting the mirrored plot below the x-axis
    # Upper part: positive values (main plot)
    # Lower part: mirrored (negative) values
    plt.plot(freq_axis_fft, -fft_magnitude_normalized, color="hotpink", alpha=0.3)  # Reflection with reduced alpha

    plt.fill_between(freq_axis_fft, 0, fft_magnitude_normalized, color="hotpink", alpha=0.6)
    # Customizing appearance
    plt.grid(False)  # Turn off the grid lines
    plt.gca().set_facecolor('black')  # Set the background color to black

    # Ensure that the plot background remains black even during saving
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Make sure the figure fills the space

    # Set the x-axis to logarithmic scale
    plt.xscale("log")

    # Remove chart title
    plt.title("", fontsize=0)  # No title
    plt.axis('off')

    # Save plot as image with black background (no transparency)
    plot_path = os.path.join(output_dir, "temp_frame.png")
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0, transparent=False)  # Explicitly set transparent=False
    plt.close()

    # Now, let's ensure OpenCV reads the image correctly
    frame = cv2.imread(plot_path)  # Read without alpha channel
    resized_frame = cv2.resize(frame, (width, height))
    video_out.write(resized_frame)

video_out.release()
os.remove(plot_path)  # Clean up temp frame
print(f"🎥 Video saved to: {video_name}")

# ===========================
# 5. Add Audio to Video
# ===========================
video_clip = VideoFileClip(video_name)
audio_clip = AudioFileClip(output_audio_path)

# Set audio to the video clip
final_video = video_clip.with_audio(audio_clip)

# Save final video with audio
final_video_path = os.path.join(output_dir, "final_video_with_audio.mp4")  # Change extension to .mp4
final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

# ===================
# Final Confirmation
# ===================
print("✅ Fourier Transform Video Generation Complete!")
print(f"Processed audio saved at: {output_audio_path}")
print(f"Final video with audio saved at: {final_video_path}")
