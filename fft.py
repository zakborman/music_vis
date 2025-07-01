import pygame
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile
import os
from pydub import AudioSegment  # MP3 support

def run_visualizer(audio_path, title, color):
    # --- SETUP ---
    pygame.init()
    screen = pygame.display.set_mode((800, 450))
    pygame.display.set_caption("Film Soundtrack Visualizer")
    clock = pygame.time.Clock()

    # --- LOAD AUDIO (MP3 → WAV conversion if needed) ---
    play_path = audio_path
    if audio_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_path)
        audio.export("temp.wav", format="wav")
        sample_rate, audio_data = wavfile.read("temp.wav")
        play_path = "temp.wav"  # Play the WAV!
    else:
        sample_rate, audio_data = wavfile.read(audio_path)

    # Convert to mono if stereo
    audio_mono = audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data
    total_samples = len(audio_mono)
    audio_duration = total_samples / sample_rate

    # --- PLAY AUDIO ---
    pygame.mixer.init(frequency=sample_rate)
    pygame.mixer.music.load(play_path)
    pygame.mixer.music.play()

    # --- FFT SETTINGS ---
    chunk_size = 1024  # Power of 2

    def get_fft(chunk):
        """Compute normalized FFT with DC offset removal and windowing (no smoothing window)."""
        chunk = chunk - np.mean(chunk)  # Remove DC offset
        window = np.hanning(len(chunk))
        chunk = chunk * window
        fft_data = np.abs(fft(chunk))[:chunk_size // 2]
        return fft_data / np.max(fft_data) if np.max(fft_data) > 0 else fft_data

    fft_frames = []
    for pos in range(0, total_samples - chunk_size, chunk_size // 4):
        chunk = audio_mono[pos : pos + chunk_size]
        fft_frames.append(get_fft(chunk))

    # --- MAIN LOOP ---
    running = True
    prev_spectrum = np.zeros(len(fft_frames[0]))
    alpha = 0.1  # Smoothing factor: 0.1 = very smooth, 0.5 = responsive

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ms_pos = pygame.mixer.music.get_pos()
        if ms_pos == -1:
            running = False
            break
        playback_time = ms_pos / 1000.0
        playback_time = min(playback_time, audio_duration)

        # Interpolate between FFT frames for smoothness
        current_sample = playback_time * sample_rate
        fft_pos = current_sample / (chunk_size // 4)
        fft_idx = int(fft_pos)
        fft_idx_next = min(fft_idx + 1, len(fft_frames) - 1)
        alpha_interp = fft_pos - fft_idx

        # Prevent index error at the end
        if fft_idx >= len(fft_frames):
            spectrum = fft_frames[-1]
        else:
            spectrum = (1 - alpha_interp) * fft_frames[fft_idx] + alpha_interp * fft_frames[fft_idx_next]

        # --- TEMPORAL SMOOTHING ---
        prev_spectrum = alpha * spectrum + (1 - alpha) * prev_spectrum

        # --- DRAW ---
        screen.fill((0, 0, 0))

        for i, mag in enumerate(prev_spectrum):
            bar_height = int(mag * 300)
            pygame.draw.rect(screen, color, (i * 8, 400 - bar_height, 6, bar_height))

        progress = playback_time / audio_duration
        pygame.draw.rect(screen, (255, 0, 0), (0, 430, int(800 * progress), 10))

        font = pygame.font.SysFont("Arial", 24)
        text = font.render(title, True, (255, 255, 255))
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    if play_path == "temp.wav":
        os.remove("temp.wav")  # Clean up

def main_menu():
    while True:
        print("\nSelect a soundtrack to visualize:")
        print("1. Mission: Impossible")
        print("2. Star Wars - Imperial March")
        print("3. Star Wars - Rey's Theme")
        print("4. Jurassic Park")
        print("5. Quit")
        choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

        if choice == "1":
            run_visualizer("lib/mission_impossible.mp3", "MISSION: IMPOSSIBLE", (255, 0, 0))  # Red
        elif choice == "2":
            run_visualizer("lib/sw_imperial_march.mp3", "STAR WARS: IMPERIAL MARCH", (255, 215, 0))
        elif choice == "3":
            run_visualizer("lib/sw_rey.mp3", "STAR WARS: REY'S THEME", (135, 206, 250))  # Light blue
        elif choice == "4":
            run_visualizer("lib/jurassic_park.mp3", "JURASSIC PARK", (255, 69, 0))  # Red-orange
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()