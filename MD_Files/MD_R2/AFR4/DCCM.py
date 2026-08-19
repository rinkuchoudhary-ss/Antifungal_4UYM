import pandas as pd
import numpy as np
import math 
import matplotlib
import matplotlib.pyplot as plt

# Ensure Times New Roman is available
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.family'] = 'Times New Roman'

# Load the output file of GROMACS covar ascii
import os
os.chdir(r"C:\Users\HP\Desktop\SilicoScientia\AntiFungal-Project3\MD_plots_R1_R2\AFR4\DCCM")
covar = pd.read_csv(r'AFR4_R1.dat', sep=' ', header=None)

# Parsing the covariance file
resnum = int(math.sqrt((len(covar.index))/3))
all_results = pd.DataFrame()
for i in range(0,resnum):
    three_step = pd.DataFrame()
    for j in range((i*resnum)*3,int(len(covar)/resnum)*(i+1),resnum):
        df = covar[j:resnum+j]
        df2 = df.reset_index(drop=True)
        three_step = pd.concat([three_step,df2], ignore_index=True, axis=1)
    
    all_results = pd.concat([all_results,three_step], ignore_index=True, axis=0)
all_results['sum'] = all_results.sum(axis=1)
a = all_results['sum'].to_numpy()
cov_matrix = a.reshape(resnum,resnum)

# Convert the covariance matrix to cross-correlation
corr = np.zeros((resnum,resnum))
for i in range(0,resnum):
    for j in range(0,resnum):
        corr[i,j] = cov_matrix[i,j]/math.sqrt(cov_matrix[i,i]*cov_matrix[j,j])

# Save the cross-correlation matrix as csv file
np.savetxt("AFR4_R1.csv", corr, delimiter=" ", fmt='%s')

# Draw the graph
file = np.loadtxt('AFR4_R1.csv')
data = np.array(file)

# Create figure with Times New Roman font
fig, ax = plt.subplots(figsize=(11,9))

# Plot the heatmap
im = ax.imshow(data, cmap=plt.cm.inferno, vmin=-1, vmax=1, origin='lower')

# Colorbar with Times New Roman
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=20)

# Set plot limits
# plt.xlim(0,360)
# plt.ylim(0,360)

# Axis labels and ticks with Times New Roman
ax.set_xlabel('Residue index', fontsize=30, fontfamily='Times New Roman')
ax.set_ylabel('Residue index', fontsize=30, fontfamily='Times New Roman')

# Tick parameters
ax.tick_params(labelsize=20)

# Ensure all tick labels use Times New Roman
for label in (ax.get_xticklabels() + ax.get_yticklabels() + 
              cbar.ax.get_yticklabels()):
    label.set_fontname('Times New Roman')

# Adjust layout and save
fig.tight_layout()
fig.savefig('AFR4_R1.jpeg', dpi=600, format='jpeg', pil_kwargs={'optimize':True, 'quality':80})