import os
from io import StringIO
from matplotlib import pyplot as plt
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Function to parse xvg files and return an array
def xvg_data_parser(input_file):
    try:
        with open(input_file, 'r') as infile:
            data = []
            lines = infile.readlines()
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue
                data.append(line)
            result = "".join(data)
            array = np.genfromtxt(StringIO(result))
            return array
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return None


# Function to select multiple graph types and return plot labels for each type
def sele_type():
    try:
        num_graphs = int(input("How many types of graphs do you want to plot? "))
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return []

    graph_types = []
    
    for i in range(num_graphs):
        print(f"Graph types: 1:gyration, 2:rmsf, 3:protein rmsd, 4:ligand rmsd, 5:hbond, 6:sasa")
        graph_type = input(f"Enter the graph type number for graph {i+1}: ")
        
        title, xlabel, ylabel = '', '', ''
        if graph_type == "1":
            title = "Radius of Gyration"
            xlabel = "Time (ps)"
            ylabel = "Radius of Gyration (nm)"
        elif graph_type == "2":
            title = "RMSF"
            xlabel = "Residue number"
            ylabel = "RMSF (nm)"
        elif graph_type == "3":
            title = "Protein RMSD"
            xlabel = "Time (ns)"
            ylabel = "Protein backbone RMSD (nm)"
        elif graph_type == "4":
            title = "Ligand RMSD"
            xlabel = "Time (ns)"
            ylabel = "Ligand RMSD (nm)"
        elif graph_type == "5":
            title = "H-Bonds"
            xlabel = "Time (ns)"
            ylabel = "Number of H-bonds"
        elif graph_type == "6":
            title = "SASA"
            xlabel = "Time (ns)"
            ylabel = "Area (nm)"
        else:
            print("Invalid input")
            return []
        
        # Get folder for each graph type
        folder = input(f"Enter the folder name for graph type {graph_type} (e.g., for {title}): ")
        
        graph_types.append((title, xlabel, ylabel, graph_type, folder))
    
    return graph_types


# Function to plot all graphs from different files on the same axes
def plot_all_graphs(x_values, y_values_list, labels, title, xlabel, ylabel, folder, colors):
    if not os.path.exists(folder):
        os.makedirs(folder)  # Ensure folder exists

    FONTSIZE = 24  # Single source of truth — change here to affect everything

    # Set ALL rcParams BEFORE creating the figure so subplots inherits them
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': FONTSIZE,
        'axes.titlesize': FONTSIZE,
        'axes.labelsize': FONTSIZE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'figure.autolayout': True,
        'axes.linewidth': 1.5,
    })

    f, ax = plt.subplots(1, figsize=(17, 10))

    cnt = 0
    print(len(y_values_list))
    for y_values, label in zip(y_values_list, labels):
        cnt += 1
        if len(y_values_list) <= cnt:
            cnt = 0
        print(cnt)
        plt.plot(x_values, y_values, label=label.replace(".xvg", ""), color=colors[cnt], linewidth=0.9)

    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)

    leg = ax.legend(loc='upper center', frameon=False, ncol=3, fontsize=FONTSIZE)
    plt.grid(False)

    # tick_params sets both tick marks AND label size in one call
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE,
                   width=5, direction="in", length=8)

    max_x_value = np.max(x_values)
    min_x_value = np.min(x_values)
    min_y_value = np.min([np.min(y) for y in y_values_list])
    max_y_value = np.max([np.max(y) for y in y_values_list])

    ax.set_xlim(min_x_value, max_x_value)
    ax.set_ylim(min_y_value, max_y_value + 0.2)  # 0.2 buffer above max

    plt.savefig(f'{folder}/{folder}.jpg', dpi=300, format='jpeg', pil_kwargs={'optimize': True, 'quality': 80})
    plt.show()


# Updated function to create CSV from multiple xvg files in the folder
def create_csv_file(output_file, combined_data, labels):
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time/Residue"] + labels)
            writer.writerows(combined_data)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


# Main function to handle graph plotting and file processing
def main():
    graph_types = sele_type()

    if not graph_types:
        print("No valid graph types selected.")
        return

    for title, xlabel, ylabel, graph_type, folder in graph_types:
        y_values_list = []
        x_values = None
        labels = []

        for file in os.listdir(folder):
            if file.endswith(".xvg"):
                file_path = os.path.join(folder, file)
                array = xvg_data_parser(file_path)

                if array is None or array.ndim != 2 or array.shape[1] < 2:
                    print(f"Skipping file {file}: Invalid or empty data")
                    continue

                x = array[:, 0]
                y = array[:, 1]

                if x_values is None:
                    x_values = x
                elif not np.array_equal(x_values, x):
                    print(f"Warning: X values in {file} differ from others. Skipping this file.")
                    continue

                y_values_list.append(y)
                labels.append(file)

        if y_values_list:
            colors = ["#0077FF", "#FF006E", "#00C853"]
            if graph_type == '5' or graph_type == '1':
                x_values = x_values / 1000
            plot_all_graphs(x_values, y_values_list, labels, title, xlabel, ylabel, folder, colors)

        if x_values is not None and y_values_list:
            combined_data = np.column_stack([x_values] + y_values_list)
            csv_file_name = os.path.join(folder, f"data_{graph_type}.csv")
            create_csv_file(csv_file_name, combined_data, labels)


if __name__ == '__main__':
    main()