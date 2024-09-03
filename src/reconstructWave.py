import numpy as np
import scipy.signal
import scipy.fft
import pdb

def short_time_fourier_transform(signal, fft_size=1024, overlap_factor=4):
    """Compute the Short-Time Fourier Transform (STFT) of a signal.

    Args:
        signal (np.ndarray): Input audio signal.
        fft_size (int): Size of the FFT window.
        overlap_factor (int): Overlap factor between windows.

    Returns:
        np.ndarray: STFT of the input signal.
    """
    hop_size = fft_size // overlap_factor
    window = scipy.signal.get_window('hann', fft_size, fftbins=False)
    stft_result = np.array([
        scipy.fft.rfft(window * signal[i:i + fft_size])
        for i in range(0, len(signal) - fft_size, hop_size)
    ])
    return stft_result

def inverse_short_time_fourier_transform(stft_matrix, overlap_factor=4):
    """Compute the inverse Short-Time Fourier Transform (ISTFT) from a complex spectrogram.

    Args:
        stft_matrix (np.ndarray): Complex STFT matrix.
        overlap_factor (int): Overlap factor used in the STFT.

    Returns:
        np.ndarray: Reconstructed time-domain signal.
    """
    fft_size = (stft_matrix.shape[1] - 1) * 2
    hop_size = fft_size // overlap_factor
    window = scipy.signal.get_window('hann', fft_size, fftbins=False)
    
    # Initialize output signal and window sum
    signal_length = stft_matrix.shape[0] * hop_size
    reconstructed_signal = np.zeros(signal_length)
    window_sum = np.zeros(signal_length)
    
    for n, start in enumerate(range(0, signal_length - fft_size, hop_size)):
        # Overlap-add method
        reconstructed_signal[start:start + fft_size] += np.real(scipy.fft.irfft(stft_matrix[n])) * window
        window_sum[start:start + fft_size] += window ** 2
    
    # Normalize by the window sum to avoid artifacts
    nonzero_indices = window_sum > 0
    reconstructed_signal[nonzero_indices] /= window_sum[nonzero_indices]
    
    return reconstructed_signal

def reconstruct_waveform_from_spectrogram(spectrogram, target_length, fft_size=1024, overlap_factor=4, iterations=8):
    """Reconstruct a waveform from a given spectrogram using Griffin-Lim algorithm.

    Args:
        spectrogram (np.ndarray): Input spectrogram (magnitude).
        target_length (int): Desired length of the output waveform.
        fft_size (int): Size of the FFT window.
        overlap_factor (int): Overlap factor used in the STFT.
        iterations (int): Number of Griffin-Lim iterations.

    Returns:
        np.ndarray: Reconstructed waveform.
    """
    # Initialize with random noise
    reconstructed_signal = np.random.rand(target_length * 2)
    
    for _ in range(iterations):
        # Compute STFT of the current estimate
        current_stft = short_time_fourier_transform(reconstructed_signal, fft_size, overlap_factor)
        
        # Reconstruct magnitude and preserve phase
        magnitude_spectrogram = spectrogram
        pdb.set_trace()

        phase_spectrogram = np.exp(1j * np.angle(current_stft[:spectrogram.shape[0], :]))
        new_stft = magnitude_spectrogram * phase_spectrogram
        
        # Inverse STFT to get time-domain signal
        reconstructed_signal = inverse_short_time_fourier_transform(new_stft, overlap_factor)
    
    return reconstructed_signal[:target_length]

