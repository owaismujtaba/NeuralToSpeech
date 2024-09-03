import os
import numpy as np
import pandas as pd
import numpy.matlib as matlib
import scipy.signal
import scipy.io.wavfile
from pynwb import NWBHDF5IO
import src.MelFilterBank as mel
import src.config as config
from multiprocessing import Pool, cpu_count
import pdb

class FeatureExtractor:
    def __init__(self, eeg_sr, audio_sr, win_length=0.05, frameshift=0.01, model_order=4, step_size=5):
        """
        Initializes the FeatureExtractor with the given parameters.

        Parameters
        ----------
        eeg_sr: int
            Sampling rate of the EEG data.
        audio_sr: int
            Sampling rate of the audio data.
        win_length: float, optional
            Length of the window in seconds (default is 0.05).
        frameshift: float, optional
            Shift in seconds after which the next window will be extracted (default is 0.01).
        model_order: int, optional
            Number of temporal contexts to include prior to and after the current window (default is 4).
        step_size: int, optional
            Number of temporal contexts to skip for each next context (default is 5).
        """
        self.eeg_sr = eeg_sr
        self.audio_sr = audio_sr
        self.win_length = win_length
        self.frameshift = frameshift
        self.model_order = model_order
        self.step_size = step_size

    @staticmethod
    def hilbert_transform(x):
        return scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]

    def extract_high_gamma(self, data):
        """
        Extracts High-Gamma frequency band envelope using the Hilbert transform.

        Parameters
        ----------
        data: array (samples, channels)
            EEG time series data.

        Returns
        ----------
        feat: array (windows, channels)
            Frequency-band feature matrix.
        """
        data = scipy.signal.detrend(data, axis=0)
        num_windows = int(np.floor((data.shape[0] - self.win_length * self.eeg_sr) / (self.frameshift * self.eeg_sr)))

        # Filter High-Gamma Band
        sos = scipy.signal.iirfilter(4, [70 / (self.eeg_sr / 2), 170 / (self.eeg_sr / 2)], btype='bandpass', output='sos')
        data = scipy.signal.sosfiltfilt(sos, data, axis=0)

        # Attenuate line noise harmonics
        for freq in [100, 150]:
            sos = scipy.signal.iirfilter(4, [(freq - 2) / (self.eeg_sr / 2), (freq + 2) / (self.eeg_sr / 2)],
                                         btype='bandstop', output='sos')
            data = scipy.signal.sosfiltfilt(sos, data, axis=0)

        # Create feature space
        data = np.abs(self.hilbert_transform(data))
        feat = np.zeros((num_windows, data.shape[1]))

        for win in range(num_windows):
            start = int(np.floor((win * self.frameshift) * self.eeg_sr))
            stop = int(np.floor(start + self.win_length * self.eeg_sr))
            feat[win, :] = np.mean(data[start:stop, :], axis=0)

        return feat

    def stack_features(self, features):
        """
        Adds temporal context to each window by stacking neighboring feature vectors.

        Parameters
        ----------
        features: array (windows, channels)
            Feature time series.

        Returns
        ----------
        feat_stacked: array (windows, feat*(2*model_order+1))
            Stacked feature matrix.
        """
        num_windows = features.shape[0] - (2 * self.model_order * self.step_size)
        feat_stacked = np.zeros((num_windows, (2 * self.model_order + 1) * features.shape[1]))

        for i in range(self.model_order * self.step_size, features.shape[0] - self.model_order * self.step_size):
            ef = features[i - self.model_order * self.step_size:i + self.model_order * self.step_size + 1:self.step_size, :]
            feat_stacked[i - self.model_order * self.step_size, :] = ef.flatten()

        return feat_stacked  

    def extract_mel_spectrogram(self, audio, audio_sr):
        """
        Extracts logarithmic mel-scaled spectrogram.

        Parameters
        ----------
        audio: array
            Audio time series.

        Returns
        ----------
        spectrogram: array (num_windows, num_filter)
            Logarithmic mel scaled spectrogram.
        """
        num_windows = int(np.floor((audio.shape[0] - self.win_length * self.audio_sr) / (self.frameshift * self.audio_sr)))
        win = scipy.signal.get_window('hann', int(self.win_length * self.audio_sr), fftbins=False)
        spectrogram = np.zeros((num_windows, int(np.floor(self.win_length * self.audio_sr / 2 + 1))), dtype='complex')

        for w in range(num_windows):
            start_audio = int(np.floor((w * self.frameshift) * self.audio_sr))
            stop_audio = int(np.floor(start_audio + self.win_length * self.audio_sr))
            a = audio[start_audio:stop_audio]
            spec = np.fft.rfft(win * a)
            spectrogram[w, :] = spec

        mfb = mel.MelFilterBank(spectrogram.shape[1], config.no_of_mel_spectrograms, audio_sr)
        spectrogram = np.abs(spectrogram)
        spectrogram = mfb.toLogMels(spectrogram).astype('float')
        

        return spectrogram

   

def process_participant(participant):
    win_length = 0.05
    frameshift = 0.01
    model_order = 4
    step_size = 5
    path_bids = config.dataset_dir
    path_output = config.features_dir
    
    print(f'Extracting features for {participant}')
    # Load data
    io = NWBHDF5IO(os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
    nwbfile = io.read()

    eeg = nwbfile.acquisition['iEEG'].data[:]
    eeg_sr = 1024
    audio = nwbfile.acquisition['Audio'].data[:]
    audio_sr = 48000
    words = np.array(nwbfile.acquisition['Stimulus'].data[:], dtype=str)
    io.close()

    channels = pd.read_csv(os.path.join(path_bids, participant, 'ieeg', f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
    channels = np.array(channels['name'])

    # Feature Extraction
    extractor = FeatureExtractor(eeg_sr=eeg_sr, audio_sr=audio_sr, win_length=win_length, frameshift=frameshift, model_order=model_order, step_size=step_size)

    feat = extractor.extract_high_gamma(eeg)
    feat = extractor.stack_features(feat)

    # Process Audio
    target_sr = 16000
    audio = scipy.signal.decimate(audio, int(audio_sr / target_sr))
    audio_sr = target_sr
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    os.makedirs(path_output, exist_ok=True)
    scipy.io.wavfile.write(os.path.join(path_output, f'{participant}_orig_audio.wav'), audio_sr, scaled)

    # Extract Spectrogram
    mel_spec = extractor.extract_mel_spectrogram(scaled, audio_sr)    
    mel_spec = mel_spec[model_order * step_size:mel_spec.shape[0] - model_order * step_size, :]

    if mel_spec.shape[0] != feat.shape[0]:
        t_len = min(mel_spec.shape[0], feat.shape[0])
        mel_spec = mel_spec[:t_len, :]
        feat = feat[:t_len, :]


    print('Saving Files')
    np.save(os.path.join(path_output, f'{participant}_feat.npy'), feat)
    np.save(os.path.join(path_output, f'{participant}_spec.npy'), mel_spec)


def extract_features_for_all_participants():
    path_bids = config.dataset_dir
    participants = pd.read_csv(os.path.join(path_bids, 'participants.tsv'), delimiter='\t')
    participant_list = participants['participant_id'].tolist()
    for participant in participant_list:
        process_participant(participant)
    # Use multiprocessing to process participants in parallel
    #with Pool(processes=config.num_jobs) as pool:
    #    pool.map(process_participant, participant_list)
