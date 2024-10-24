# %%
import pandas as pd
import numpy as np

winding = pd.read_csv('data/Winding-lab-fly-stocks_2023-09-14.csv')

clean_brain = pd.read_csv('data/1_Brain-1pair_clean_145splitGAL4s.csv')
clean_brain_VNC = pd.read_csv('data/2_Brain-1pair_dirty-VNC_111splitGAL4s.csv')
pairs2_brain_lineage = pd.read_csv('data/3_Brain-2pairs_same-lineage_98splitGAL4s.csv')
pairs2_brain_nonlineage = pd.read_csv('data/4_Brain-2pairs_different-lineage_115splitGAL4s.csv')

# %%
# check for lines

# clean brain lines
boolean = winding['ID'].isin(clean_brain.Line.values)
winding.loc[boolean, 'Num_Brain'] = [1]*len(winding.loc[boolean, 'Num_Brain'])
winding.loc[boolean, 'VNC_expression'] = [False]*len(winding.loc[boolean, 'VNC_expression'])


reg = '_([A-Za-z\d]+)'

# clean brain lines with VNC expression
clean_brain_VNC['extracted'] = clean_brain_VNC['Line']

mask = clean_brain_VNC['Line'].str.contains('_')
clean_brain_VNC.loc[mask, 'extracted'] = clean_brain_VNC.loc[mask, 'Line'].str.extract(reg)[0]

boolean = winding['ID'].isin(clean_brain_VNC.extracted.values)
winding.loc[boolean, 'Num_Brain'] = [1]*len(winding.loc[boolean, 'Num_Brain'])
winding.loc[boolean, 'VNC_expression'] = [True]*len(winding.loc[boolean, 'VNC_expression'])

# 2pair brain lines in lineage
pairs2_brain_lineage['extracted'] = pairs2_brain_lineage['Line']

mask = pairs2_brain_lineage['Line'].str.contains('_')
pairs2_brain_lineage.loc[mask, 'extracted'] = pairs2_brain_lineage.loc[mask, 'Line'].str.extract(reg)[0]

boolean = winding['ID'].isin(pairs2_brain_lineage.extracted.values)
winding.loc[boolean, 'Num_Brain'] = [2]*len(winding.loc[boolean, 'Num_Brain'])
winding.loc[boolean, 'VNC_expression'] = [np.nan]*len(winding.loc[boolean, 'VNC_expression'])
winding.loc[boolean, 'Notes_expression'] = ['same lineage in brain']*len(winding.loc[boolean, 'Notes_expression'])

# 2pair brain lines in nonlineage
pairs2_brain_nonlineage['extracted'] = pairs2_brain_nonlineage['Line']

mask = pairs2_brain_nonlineage['Line'].str.contains('_')
pairs2_brain_nonlineage.loc[mask, 'extracted'] = pairs2_brain_nonlineage.loc[mask, 'Line'].str.extract(reg)[0]

boolean = winding['ID'].isin(pairs2_brain_nonlineage.extracted.values)
winding.loc[boolean, 'Num_Brain'] = [2]*len(winding.loc[boolean, 'Num_Brain'])
winding.loc[boolean, 'VNC_expression'] = [np.nan]*len(winding.loc[boolean, 'VNC_expression'])
winding.loc[boolean, 'Notes_expression'] = ['different lineage in brain']*len(winding.loc[boolean, 'Notes_expression'])

# %%
# export fly-stocks csv

winding.to_csv('data/Modified_Winding-lab-fly-stocks_2023-09-14.csv')