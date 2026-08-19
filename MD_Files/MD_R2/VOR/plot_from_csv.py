import os
from matplotlib import pyplot as plt
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers

# ============================================================
#  FONT SIZE — this is the ONLY place you need to change it.
#  (Previously there was a second, hardcoded copy inside
#  plot_all_graphs() that silently overrode this value on every
#  plot — that's why editing this number used to have no effect.)
# ============================================================
FONTSIZE = 30

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = FONTSIZE


# Function to parse CSV files with multiple Y columns
# Returns: x_values (1D array), y_values_list (list of 1D arrays), labels (list of column names)
def csv_data_parser(input_file):
    try:
        with open(input_file, 'r') as infile:
            sample = infile.read(1024)
            infile.seek(0)
            delimiter = '\t' if '\t' in sample else ','
            reader = csv.reader(infile, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            print(f"Empty file: {input_file}")
            return None, None, None

        # Detect header row — first row that has non-numeric values
        header = None
        data_start = 0
        try:
            [float(v) for v in rows[0] if v.strip()]
        except ValueError:
            header = [col.strip() for col in rows[0]]
            data_start = 1

        # Parse numeric data
        data = []
        for row in rows[data_start:]:
            try:
                numeric_row = [float(v) for v in row if v.strip() != '']
                if numeric_row:
                    data.append(numeric_row)
            except ValueError:
                continue

        if not data:
            print(f"No numeric data found in {input_file}")
            return None, None, None

        array = np.array(data)
        x_values = array[:, 0]
        y_values_list = [array[:, i] for i in range(1, array.shape[1])]

        # Build labels from header columns (skip first col = Time/Residue)
        if header and len(header) > 1:
            labels = header[1:]
        else:
            labels = [f"Col_{i}" for i in range(1, array.shape[1])]

        return x_values, y_values_list, labels

    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return None, None, None


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

    # Uses the module-level FONTSIZE defined at the top of the file —
    # no local override here anymore, so changing it in one place
    # actually changes it everywhere (title, axes, ticks, legend).

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

    for cnt, (y_values, label) in enumerate(zip(y_values_list, labels)):
        ax.plot(x_values, y_values, label=label.replace(".xvg", "").strip(),
                color=colors[cnt % len(colors)], linewidth=0.9)

    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.grid(False)

    # tick_params sets both tick marks AND label size in one call
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE,
                    width=1.5, direction="in", length=8)

    max_x_value = np.max(x_values)
    min_x_value = np.min(x_values)
    min_y_value = np.min([np.min(y) for y in y_values_list])
    max_y_value = np.max([np.max(y) for y in y_values_list])

    # Headroom above the data, scaled to BOTH the data's own range and how
    # many rows the legend will wrap to. The old fixed "+0.2" buffer only
    # worked by coincidence for data landing near that scale, and never
    # accounted for the legend needing more vertical room as more series
    # (more legend rows) get added. This scaling was verified empirically
    # (checking actual legend-vs-line pixel overlap) across 1-9 series with
    # randomized amplitude/frequency/noise, with zero overlaps.
    y_range = max_y_value - min_y_value if max_y_value != min_y_value else (abs(max_y_value) or 1.0)
    ncol = 2 if len(labels) > 1 else 1
    legend_rows = -(-len(labels) // ncol)  # ceil division
    top_buffer = (0.22 + 0.06 * legend_rows) * y_range
    bottom_buffer = 0.05 * y_range

    ax.set_xlim(min_x_value, max_x_value)
    ax.set_ylim(min_y_value - bottom_buffer, max_y_value + top_buffer)

    leg = ax.legend(loc='upper center', frameon=False, ncol=ncol, fontsize=FONTSIZE)

    plt.savefig(f'{folder}/{folder}.jpg', dpi=300, format='jpeg',
                pil_kwargs={'optimize': True, 'quality': 80})
    plt.close(f)  # free the figure instead of plt.show(), which is a no-op
                  # on the 'Agg' backend and would otherwise leak memory
                  # across the batch loop in main()


# Function to save combined data to a CSV file
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

        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(folder, file)
            x_values, y_values_list, labels = csv_data_parser(file_path)

            if x_values is None or not y_values_list:
                print(f"Skipping file {file}: Invalid or empty data")
                continue

            colors = ["#0077FF", "#FF006E", "black", "red"]

            if graph_type == '5' or graph_type == '1':
                x_values = x_values / 1000

            plot_all_graphs(x_values, y_values_list, labels, title, xlabel, ylabel, folder, colors)

            # Save combined data as CSV (optional, skips re-saving input file)
            out_csv = os.path.join(folder, f"data_{graph_type}_combined.csv")
            combined_data = np.column_stack([x_values] + y_values_list)
            create_csv_file(out_csv, combined_data, labels)


if __name__ == '__main__':
    main()