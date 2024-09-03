import os
from pathlib import Path
import src.config as config
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import src.config as config
import pandas as pd

def plot_history():
    print('*****************Plotting History******************')

    folder = 'correlations'
    destination = Path(config.current_dir, 'Results')
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
    plt.tight_layout()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig(Path(destination, 'history.png'), dpi=600)
    plt.clf()
    


def plot_correlation():
    print('*****************Plotting Correlations******************')

    folder = Path(config.current_dir, 'correlations')
    destination = Path(config.current_dir, 'Results')
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
    std_per_subject = np.sqrt(np.mean(std_errors**2, axis=1) / num_folds)  # Shape: (10,)

    sorted_indices = np.argsort(mean_per_subject)
    sorted_mean_per_subject = mean_per_subject[sorted_indices]
    sorted_std_per_subject = std_per_subject[sorted_indices]
    sorted_mean_correlations = mean_correlations[sorted_indices]
    sorted_dot_colors = sns.color_palette("coolwarm", n_colors=num_subjects)

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_colors = sns.color_palette("viridis", n_colors=num_subjects)

    bars = ax.bar(range(num_subjects), sorted_mean_per_subject, yerr=sorted_std_per_subject, capsize=8, color=[bar_colors[i] for i in range(num_subjects)], edgecolor='black', alpha=0.8, label='Mean Correlation')

    for i in range(num_subjects):
        ax.plot(np.full(num_folds, i), sorted_mean_correlations[i], 's', color=sorted_dot_colors[i], alpha=0.7, label=f'Folds (Subject {i + 1})', markersize=8, linestyle='None')

  
    ax.set_ylabel('Correlation', fontsize=14)
    ax.set_xticks(range(num_subjects))
    ax.set_xticklabels([f'sub-{names[i]}' for i in sorted_indices], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0.8,1])
    fig.tight_layout()

    # Show the plot
    plt.savefig(Path(destination, 'correlations.png'), dpi=600)
    plt.clf()



def plot_stgis():
    print('*****************Plotting STGIS******************')
    folder = Path(config.current_dir, 'correlations')
    destination = Path(config.current_dir, 'Results')
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
    sorted_dot_colors = sns.color_palette("coolwarm", n_colors=num_subjects)
    
    # Creating box plots
    fig, ax = plt.subplots(figsize=(14, 7))
    box_colors = sns.color_palette("viridis", n_colors=num_subjects)
    
    meanprops = dict(marker='s', markerfacecolor='white', markeredgecolor='black', markersize=7)
    box_data = [sorted_mean_correlations[i] for i in range(num_subjects)]
    
    box = ax.boxplot(box_data, patch_artist=True, notch=False, showmeans=True, 
                     meanprops=meanprops, showfliers=False)

    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # Plot individual data points (after boxplot so they appear in front)
    for i in range(num_subjects):
        y = sorted_mean_correlations[i]
        x = np.random.normal(i + 1, 0.04, size=len(y))  # Add some jitter to the x-axis
        ax.scatter(x, y, color=sorted_dot_colors[i], alpha=0.7, s=20, edgecolor='black', zorder=3)

    ax.set_ylabel('STGI', fontsize=14)
    ax.set_xticks(range(1, num_subjects + 1))
    ax.set_xticklabels([f'sub-{names[i]}' for i in sorted_indices], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([0.45, 0.65])
    fig.tight_layout()

    # Save the plot
    plt.savefig(Path(destination, 'stgis.png'), dpi=600)
    plt.clf()
