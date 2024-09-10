from src.feature_extractor import extract_features_for_all_participants
from src.audio_constructor import AudioReconstructor
import src.config as config
from src.vis import plot_correlation, plot_stgis, plot_history, plot_spectrograms
import pdb
import warnings
warnings.filterwarnings('ignore')

if config.extract_features:
    extract_features_for_all_participants()

if config.construct:
    reconstruct = AudioReconstructor()
    reconstruct.reconstruct()

if config.visualization:
    plot_correlation()
    plot_stgis()
    plot_history()
    plot_spectrograms()
