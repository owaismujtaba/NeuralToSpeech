from src.feature_extractor import extract_features_for_all_participants
from src.audio_constructor import AudioReconstructor
import src.config as config
from src.vis import plot_correlation, plot_stgis, plot_history
import pdb
import warnings
import torch
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

'''
import sys
sys.path.append('tacotron/')
from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.layers import TacotronSTFT, STFT
from tacotron2.audio_processing import griffin_lim
from tacotron2.train import load_model

hparms = create_hparams()
hparms.sampling_rate=16000
pdb.set_trace()
checkpoint_path = 'tacotron2_statedict.pt'
model = load_model(hparms)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
'''