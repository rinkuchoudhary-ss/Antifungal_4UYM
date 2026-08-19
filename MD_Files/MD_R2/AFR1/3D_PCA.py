import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing the 3D plotting module

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

def read_xvg(filename):
    """Reads an .xvg file and returns the data."""
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Filter out lines that are comments or metadata
    data_lines = [line for line in lines if not (line.startswith('#') or line.startswith('@'))]

    # Split lines into columns and convert to float
    data = np.array([list(map(float, line.strip().split())) for line in data_lines])

    return data

def plot_3d_scatter(folder_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # Creating a 3D subplot
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    cnt = 0
    colors = ["#0077FF", "#FF006E", "#FFEA00"]
    for i, file_name in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        name = file_name.removesuffix(".xvg")
        if os.path.isfile(file_path) and file_name.endswith(".xvg"):
            cnt += 1
            data = read_xvg(file_path)
            secnme = name.removeprefix("2")

            secondfile = "3"+secnme+".xvg"
            file_path1 = os.path.join(folder_path,secondfile)
            print(secondfile)
            data1 = read_xvg(file_path1)


            pca1 = data[:, 0]
            pca2 = data[:, 1]
            pca3 = data1[:, 1]  # Assuming you have a third PCA component

            # Choose different color for each file
            color = colors[i % len(colors)]

            ax.scatter(pca1, pca2, pca3, color=color, label=secnme, s = 3.5)


            num = os.listdir(folder_path)
            numb = len(num)
            num1 = numb//2
            if(cnt == num1):
                break


    ax.set_xlabel('PCA Component 1')
    ax.set_xlim()
    ax.set_ylabel('PCA Component 2')
    ax.set_ylim(-10,20)
    ax.set_zlabel('PCA Component 3')  # Adjust for your third PCA component
    ax.set_zlim()
    ax.set_title('Combined 3D PCA Plot')
    ax.set_box_aspect(None, zoom=0.85)
    ax.legend(loc = 'upper right', ncol=2, borderaxespad=7)
    ax.grid(True)

    save_path = os.path.join(folder_path, "pca_combine_3d.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Get folder path from user input
folder_path = input("Enter the folder path: ")
plot_3d_scatter(folder_path)
