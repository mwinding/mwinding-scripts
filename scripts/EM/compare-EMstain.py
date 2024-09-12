# %%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

crick_path = 'data/EM_data/Crick-stain/'
superfly_path = 'data/EM_data/Superfly/'

def max_intensity(path):
    data = pd.read_csv(path)
    return max(data.Gray_Value.values)

def mean_intensity(path):
    data = pd.read_csv(path)
    return np.mean(data.Gray_Value.values)

# Function to loop through all csv files in the synapse folder and collect max intensities
def collect_max_intensities(folder_path, condition, intensities=[]):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            value = max_intensity(file_path)
            intensities.append([condition, value])
    return intensities

# Function to loop through all csv files in the synapse folder and collect max intensities
def collect_mean_intensities(folder_path, condition, intensities=[]):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            value = mean_intensity(file_path)
            intensities.append([condition, value])
    return intensities

# %%
#######
# synapses
crick_syn_path = crick_path + 'synapse'
superfly_syn_path = superfly_path + 'synapse'

# Execute the function and print the results
max_intensities = collect_max_intensities(crick_syn_path, 'megametal')
max_intensities = collect_max_intensities(superfly_syn_path, 'superfly', max_intensities)

df_syn = pd.DataFrame(max_intensities, columns = ['stain', 'pixel-intensity'])

######
# membranes
crick_mem_path = crick_path + 'membrane'
superfly_mem_path = superfly_path + 'membrane'

# Execute the function and print the results
max_intensities = collect_max_intensities(crick_mem_path, 'megametal')
max_intensities = collect_max_intensities(superfly_mem_path, 'superfly', max_intensities)

df_mem = pd.DataFrame(max_intensities, columns = ['stain', 'pixel-intensity'])

######
# synapses
crick_cyto_path = crick_path + 'cyto'
superfly_cyto_path = superfly_path + 'cyto'

# Execute the function and print the results
mean_intensities = collect_mean_intensities(crick_cyto_path, 'megametal')
mean_intensities = collect_mean_intensities(superfly_cyto_path, 'superfly', mean_intensities)

df_cyto = pd.DataFrame(mean_intensities, columns = ['stain', 'pixel-intensity'])

# %%
# plot results
fig, axs = plt.subplots(1,3,sharey=True)
# Create a barplot with seaborn
ax = axs[0] 
sns.barplot(x='stain', y='pixel-intensity', data=df_syn, ax=ax)
ax.set_title('Max pixel intensity \nat synapse')
ax.set_xlabel('Stain Type')
ax.set_ylabel('Pixel Intensity')

ax = axs[1] 
sns.barplot(x='stain', y='pixel-intensity', data=df_mem, ax=ax)
ax.set_title('Max pixel intensity \nat membrane')
ax.set_xlabel('Stain Type')
ax.set_ylabel('')

ax = axs[2] 
sns.barplot(x='stain', y='pixel-intensity', data=df_cyto, ax=ax)
ax.set_title('Mean pixel intensity \nin cytoplasm')
ax.set_xlabel('Stain Type')
ax.set_ylabel('')

# Rotate x-axis labels and set y-axis limit for all subplots
for ax in axs:
    ax.tick_params(axis='x', rotation=45)

axs[0].set_ylim(0, 65535)

# %%
