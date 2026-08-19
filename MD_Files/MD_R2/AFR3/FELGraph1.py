import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from numpy import linspace

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

def plot_fel(ax, file_path, title):
    # Load data
    x, y, z = np.loadtxt(file_path).T

    # Create meshgrid
    X, Y = np.unique(x), np.unique(y)
    xi = linspace(min(X), max(X), len(X))
    yi = linspace(min(Y), max(Y), len(Y))
    xi, yi = np.meshgrid(xi, yi)

    X1, Y1 = np.meshgrid(X, Y)

    # Interpolate data on the grid
    Z = griddata((x, y), z, (X1, Y1), method='cubic')

    # Plot the surface
    surf = ax.plot_surface(X1, Y1, Z, rstride=1, cstride=1, alpha=1, cmap=cm.jet, linewidth=0.0, antialiased=3)

    # Contour plot
    cset = ax.contourf(X1, Y1, Z, zdir='z', offset=-0, cmap=cm.jet, antialiased=3, vmin=0, vmax=18)

    # Add axis labels
    ax.set_xlabel('PC1', fontsize=16, labelpad=4)
    ax.set_ylabel('PC2', fontsize=16, labelpad=8)
    ax.set_zlabel('Free Energy', fontsize=16, labelpad=10)

    # Add title at the bottom
    ax.text2D(0.5, -0.19, title, ha='center', va='bottom', fontsize=20, transform=ax.transAxes)

    return surf, cset

def save_all_plots(folder_path, save_folder, cols):
    # Create the output folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # List all files in the folder
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Calculate the number of rows needed based on the number of files and columns
    rows = -(-len(file_list) // cols)  # Equivalent to ceil(len(file_list) / cols)

    # Create a figure with subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=300, subplot_kw={'projection': '3d'})
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # Plot FEL for each file in the folder and save as PNG
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        ax = axs[i]
        title = (os.path.splitext(os.path.basename(file_path))[0])
        surf, cset = plot_fel(ax, file_path, title)

        # Create a colorbar for each subplot
        cbar = fig.colorbar(cset, ax=ax, shrink=0.6, aspect=20, extend='neither', ticks=range(0, 21, 2), location="left")
        cbar.ax.yaxis.set_ticks_position('right')

    # Add common axis labels for all subplots
    fig.text(0.5, 0.04, ' ', ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, ' ', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.text(0.96, 0.5, ' ', ha='center', va='center', rotation='vertical', fontsize=12)

    # Hide any empty subplots
    for i in range(len(file_list), len(axs)):
        axs[i].axis('off')

    # Save the combined plot as a PNG file
    save_path = os.path.join(save_folder, "combined_FEL_plot1.png")
    plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()

# Input folder path
folder_path = input("Enter the folder path: ")

# Output folder path to save plots
save_folder = input("Enter the folder path to save plots: ")

# Specify the number of columns
cols = int(input("Enter the number of columns: "))

# Call the function to save all plots in a single image
save_all_plots(folder_path, save_folder,cols)