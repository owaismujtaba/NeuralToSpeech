import os
from pathlib import Path
import src.config as config
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import src.config as config
import pandas as pd
import pdb
fontsize = 16


def plot_history():
    print('*****************Plotting History******************')

    folder = 'Complex'
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.csv') and item.startswith('sub')]
    names = []
    for item in files:
        name = str(item).split('/')[-1].split('.')[0]
        names.append(name)
        data = pd.read_csv(item)
        plt.plot(data['val_loss'])
    
    plt.legend(names)
    plt.xlabel('Epochs', fontsize=fontsize+2, fontweight='bold')
    plt.ylabel('Mean Squared Error', fontsize=fontsize+2, fontweight='bold')
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')

    plt.tight_layout()
   
    plt.savefig(Path(destination, 'history.png'), dpi=600)
    plt.clf()
    
def plot_spectrograms():
    start, end = 1100, 1600
    predicted_neuro = np.load('NeuroIncept/predicted.npy')
    predicted_FCN = np.load('FCN/predicted.npy')
    predicted_CNN = np.load('CNN/predicted.npy')
    original = np.load('NeuroIncept/original.npy')

    num_samples_per_time = 9
    num_time_points = original.shape[1] // num_samples_per_time  # Adjusted time points
    time_adjusted_original = original[:, :num_time_points * num_samples_per_time]
    time_adjusted_original = time_adjusted_original.reshape(original.shape[0], num_time_points, num_samples_per_time).mean(axis=2)
    time_adjusted_FCN = predicted_FCN[:, :num_time_points * num_samples_per_time]
    time_adjusted_FCN = time_adjusted_FCN.reshape(predicted_FCN.shape[0], num_time_points, num_samples_per_time).mean(axis=2)
    time_adjusted_CNN = predicted_CNN[:, :num_time_points * num_samples_per_time]
    time_adjusted_CNN = time_adjusted_CNN.reshape(original.shape[0], num_time_points, num_samples_per_time).mean(axis=2)
    time_adjusted_neuro = predicted_neuro[:, :num_time_points * num_samples_per_time]
    time_adjusted_neuro = time_adjusted_neuro.reshape(original.shape[0], num_time_points, num_samples_per_time).mean(axis=2)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)

    im = ax[0, 0].imshow(time_adjusted_original[start:end].T, aspect='auto', origin='lower', cmap='viridis')
    ax[0, 0].set_title('Original Log Mel Spectrogram', fontsize=fontsize, fontweight='bold')

    ax[0, 1].imshow(time_adjusted_neuro[start:end].T, aspect='auto', origin='lower', cmap='viridis')
    ax[0, 1].set_title('NeuroIncept Decoder', fontsize=fontsize, fontweight='bold')

    ax[1, 0].imshow(time_adjusted_CNN[start:end].T, aspect='auto', origin='lower', cmap='viridis')
    ax[1, 0].set_title('CNN', fontsize=fontsize, fontweight='bold')

    ax[1, 1].imshow(time_adjusted_FCN[start:end].T, aspect='auto', origin='lower', cmap='viridis')
    ax[1, 1].set_title('FCN', fontsize=fontsize, fontweight='bold')

    for i in range(2):
        for j in range(2):
                ax[i, j].tick_params(labelleft=False, labelbottom=False)
    labels = ['a)', 'b)', 'c)', 'd)']  

    for i, ax in enumerate(ax.flat):
        ax.plot([1, 2, 3], [1, 4, 9])  
        ax.text(0.0, 1.1, labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', 
                verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    destination = Path(config.current_dir, 'Images')
    plt.savefig(Path(destination, 'spectrogram_comparison.png'), dpi=600)
    plt.clf()



def plot_correlation():
    print('*****************Plotting Correlations******************')

    folder = Path(config.current_dir, 'Complex')
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.npy') and 'corr' in item]
    num_folds = config.num_folds
    num_subjects = 10
    data = []
    names = []
    for file in files:
        temp = np.load(file)
        data.append(temp)
        names.append(str(file).split('/')[-1].split('.')[0].split('-')[1].split('_')[0])
    data = np.array(data)

    
    mean_correlations = np.mean(data, axis=2)  # Shape: (10, 10)
    std_errors = np.std(data, axis=2)  # Shape: (10, 10)

    mean_per_subject = np.mean(mean_correlations, axis=1)  # Shape: (10,)
    std_per_subject = np.std(mean_correlations, axis=1)  # Shape: (10,)
    sorted_indices = np.argsort(mean_per_subject)
    sorted_mean_per_subject = mean_per_subject[sorted_indices]
    sorted_std_per_subject = std_per_subject[sorted_indices]
    sorted_mean_correlations = mean_correlations[sorted_indices]
    sorted_dot_colors = sns.color_palette("coolwarm", n_colors=num_subjects)
    print(sorted_mean_per_subject)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_colors = sns.color_palette("viridis", n_colors=num_subjects)

    bars = ax.bar(range(num_subjects), sorted_mean_per_subject, yerr=sorted_std_per_subject, capsize=8, color=[bar_colors[i] for i in range(num_subjects)], edgecolor='black', alpha=0.8, label='Mean Correlation')

    for i in range(num_subjects):
        ax.plot(np.full(num_folds, i), sorted_mean_correlations[i], '.', color=sorted_dot_colors[i], alpha=0.7, label=f'Folds (Subject {i + 1})', markersize=8, linestyle='None')

    ax.set_ylabel('Correlation', fontsize=16, fontweight='bold')
    ax.set_xticks(range(num_subjects))
    ax.set_xticklabels([f'sub-{names[i]}' for i in sorted_indices], fontsize=14, fontweight='bold')
    #ax.set_yticklabels(ax.get_yticks(), fontsize=14, fontweight='bold')

    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0.8,0.95])
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')
    fig.tight_layout()

    # Show the plot
    plt.savefig(Path(destination, 'correlations.png'), dpi=600)
    plt.clf()

def plot_stgis():
    print('*****************Plotting STGIS******************')
    folder = Path(config.current_dir, 'Complex')
    destination = Path(config.current_dir, 'Images')
    os.makedirs(destination, exist_ok=True)
    
    files = os.listdir(folder)
    files = [Path(folder, item) for item in files if item.endswith('.npy') and 'stgi' in item]
    num_folds = config.num_folds
    num_subjects = 10
    data = []
    names = []
    for file in files:
        temp = np.load(file)
        data.append(temp)
        names.append(str(file).split('/')[-1].split('.')[0].split('-')[1].split('_')[0])
    data = np.array(data)
    
    mean_correlations = np.mean(data, axis=2)  # Shape: (10, 10)
    std_errors = np.std(data, axis=2)  # Shape: (10, 10)

    mean_per_subject = np.mean(mean_correlations, axis=1)  # Shape: (10,)
    std_per_subject = np.sqrt(np.mean(std_errors**2, axis=1) / num_folds)  # Shape: (10,)

    sorted_indices = np.argsort(mean_per_subject)
    sorted_mean_correlations = mean_correlations[sorted_indices]
    print(np.mean(sorted_mean_correlations, axis=1))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    meanprops = dict(marker='s', markerfacecolor='white', markeredgecolor='black', markersize=7)
    box_data = [sorted_mean_correlations[i] for i in range(num_subjects)]
    
    box = ax.boxplot(box_data, patch_artist=False, notch=False, showmeans=True, 
                     meanprops=meanprops, showfliers=False)

    
    
    for i in range(num_subjects):
        y = sorted_mean_correlations[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))  # Add some jitter to the x-axis
        ax.scatter(x, y, color='black', alpha=0.7, s=20, edgecolor='black', zorder=3)

    ax.set_ylabel('STGI', fontsize=16, fontweight='bold')
   
    labels = [f'sub-{names[i]}' for i in sorted_indices]
    print(labels)
    ax.set_xticklabels(labels, fontsize=fontsize, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.ylim([0.45, 0.57])
    fig.tight_layout()
    plt.xticks(fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, fontweight='bold')

    # Save the plot
    plt.savefig(Path(destination, 'stgis.png'), dpi=600)
    plt.clf()
