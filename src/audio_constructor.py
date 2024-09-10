import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import src.reconstructWave as rW
import src.MelFilterBank as mel
import src.config as config
from src.models import NeuroInceptDecoder, FCN, CNN
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from pathlib import Path
from scipy.signal import stft

import pdb

class AudioReconstructor:
    def __init__(self,  num_random=1000, win_length=0.05, frame_shift=0.01, audio_sr=16000):
        print("Initializing AudioReconstructor...")
        self.num_random = num_random
        self.win_length = win_length
        self.frame_shift = frame_shift
        self.audio_sr = audio_sr
        self.mel_filter_bank = None

    def create_audio(self, spectrogram):
        print("Creating audio from spectrogram...")
        
        

    def evaluate_fold(self,reconstructed_spectrogram, y_test):
        print("Evaluating fold...")
        
        correlations = np.zeros(y_test.shape[1])
        for spec_bin in range(y_test.shape[1]):
            r, _ = pearsonr(y_test[:, spec_bin], reconstructed_spectrogram[:, spec_bin])
            correlations[spec_bin] = r
        return correlations

    def compute_random_baseline(self, spectrogram):
        print("Computing random baseline correlations...")

        # Initialize the array to hold correlation results
        num_time_shifts = self.num_random
        num_frequency_bins = spectrogram.shape[1]
        random_correlations = np.zeros((num_time_shifts, num_frequency_bins))

        # Number of time frames in the spectrogram
        num_time_frames = spectrogram.shape[0]

        # Compute random correlations
        for i in range(num_time_shifts):
            # Randomly select a shift point within the middle 80% of the time axis
            shift_point = np.random.randint(int(num_time_frames * 0.1), int(num_time_frames * 0.9))
            
            # Circularly shift the spectrogram along the time axis
            shifted_spectrogram = np.roll(spectrogram, shift=-shift_point, axis=0)

            # Compute correlations between original and shifted spectrogram for each frequency bin
            try:
                for bin_idx in range(num_frequency_bins):
                    # Compute Pearson correlation for each frequency bin
                    original_bin = spectrogram[:, bin_idx]
                    shifted_bin = shifted_spectrogram[:, bin_idx]

                    # Ensure both arrays have the same shape
                    if len(original_bin) == len(shifted_bin):
                        correlation, _ = pearsonr(original_bin, shifted_bin)
                        random_correlations[i, bin_idx] = correlation
            except:
                pdb.set_trace()
        return random_correlations


    def reconstruct(self):
        print("Starting reconstruction process...")
        feature_path = config.features_dir
        result_path = config.results_dir
        participants = ['sub-%02d' % i for i in range(1, 11)]
        
        destination = Path(config.current_dir, 'CNN')
        os.makedirs(destination, exist_ok=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        losses = []
        mean_correlations = []
        stgis = []
        stds_corr = []
        stds_stgi = []
        
        for p_idx, participant in enumerate(participants):
            print(f'Processing participant {participant}...')

            spectrograms = np.load(os.path.join(feature_path, f'{participant}_spec.npy'))
            features = np.load(os.path.join(feature_path, f'{participant}_feat.npy'))
            
            print("Normalizing data...")
            features = self._normalize_data(features)

            X, y = features, spectrograms 
            x_train, x_test, y_train, y_test = train_test_split(X, y)

            x_train = x_train.reshape(x_train.shape[0], 9, -1)
            x_test = x_test.reshape(x_test.shape[0], 9, -1)


            model= NeuroInceptDecoder(
                input_shape=(x_train.shape[1], x_train.shape[2]),
                output_shape= y_train.shape[1]
            )
            self.estimator = model.build_model()
            #self.estimator = model
            self.estimator.compile(optimizer='adam', loss='mse')
            
            history = self.estimator.fit(x_train, y_train, 
                               batch_size=config.batch_size, epochs=config.epochs, 
                               validation_data=(x_test, y_test), 
                               callbacks=[early_stopping]
            )
            pdb.set_trace()
            self.save_spectrograms(destination,features, spectrograms)

            reconstructed_spectrograms = self.estimator.predict(x_test)
            pdb.set_trace()

            correlations, stgi = self.eveluate_on_random_folds(reconstructed_spectrograms, y_test)
            
            print(f'{participant} has mean coorelation of {np.mean(correlations)}, stgi {np.mean(stgi)}')
            
            name = Path(destination, participant)
            np.save(f'{name}_corr.npy', correlations)
            np.save(f'{name}_stgi.npy', stgi)
            
            losses.append(history.history['val_loss'][-1])
            stgis.append(np.mean(np.mean(stgi, axis=1)))
            mean_correlations.append(np.mean(np.mean(correlations, axis=1)))
            stds_corr.append(np.std(np.mean(correlations, axis=1)))
            stds_stgi.append(np.std(np.mean(stgi, axis=1)))
            history = pd.DataFrame(history.history)
            history.to_csv(Path(destination, f'{participant}.csv'))
            

        results = {'participant': participants, 
                   'MSE': losses, 
                   'correlations': mean_correlations,
                   'std_corr': stds_corr,
                   'stgi': stgis,
                   'std_stgi': stds_stgi
        }
        results = pd.DataFrame(results)
        # Save evaluation results
        print("Saving evaluation results...")
        results.to_csv(Path(destination, 'results.csv'))
    
    
    def save_spectrograms(self,destination,  features, spectrograms):
        features = features.reshape(features.shape[0], 9, -1)
        #destination = Path(config.current_dir, 'Reconstructed')
        #os.makedirs(destination, exist_ok=True)
        print('Saving original and reconstructed in ', destination)
        predicted = self.estimator.predict(features)
        np.save(Path(destination, 'predicted.npy'), predicted)
        np.save(Path(destination, 'original.npy'), spectrograms)

    
    
    def _normalize_data(self, features):
        print("Normalizing features using training data statistics...")
        sclaer = StandardScaler()
        features = sclaer.fit_transform(features)
        return features
        
    def calculate_stgi(self, original_spectrograms, predicted_spectrograms, threshold=0):
        """
        Calculate the Spectro-Temporal Glimpsing Index (STGI) for each pair of original and predicted spectrograms.

        Parameters:
        - original_spectrograms (np.ndarray): Array of 2D matrices representing original Mel spectrograms (shape: num_samples x time x frequency).
        - predicted_spectrograms (np.ndarray): Array of 2D matrices representing predicted Mel spectrograms (shape: num_samples x time x frequency).
        - threshold (float): SNR threshold to determine glimpses. Default is 0 dB.

        Returns:
        - stgis (list of floats): List of STGI values for each spectrogram pair.
        """
        epsilon = 1e-10
        num_samples = original_spectrograms.shape[0]
        stgis = []

        for index in range(num_samples):
            # Ensure non-zero values to avoid issues in log computation
            original_spectrogram = np.maximum(original_spectrograms[index], epsilon)
            predicted_spectrogram = np.maximum(predicted_spectrograms[index], epsilon)

            # Calculate SNR for each time-frequency unit
            snr_matrix = 10 * np.log10(original_spectrogram / (predicted_spectrogram - original_spectrogram + epsilon))

            # Apply the threshold to identify glimpses
            glimpse_matrix = (snr_matrix > threshold).astype(int)

            # Calculate STGI
            stgi = np.sum(glimpse_matrix) / glimpse_matrix.size
            stgis.append(stgi)

        return stgis
    

    def eveluate_on_random_folds(self, reconstructed_spectrograms, y_test):
        n_folds = config.num_folds
        correlations = []
        stgis = []
        for index in range(n_folds):
            indicies = np.random.randint(0, reconstructed_spectrograms.shape[0], size=1000)
            sample_spectrograms = reconstructed_spectrograms[indicies]
            y = y_test[indicies]
            correlation = self.evaluate_fold(sample_spectrograms, y)
            correlations.append(correlation)
            stgi = self.calculate_stgi(y, sample_spectrograms)
            stgis.append(stgi)
        
        return np.array(correlations), np.array(stgis)



