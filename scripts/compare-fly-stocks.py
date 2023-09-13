# %%
import pandas as pd
import numpy as np

winding = pd.read_csv('data/Winding-lab-fly-stocks_2023-09-05.csv')
zlatic = pd.read_csv('data/Zlatic-lab-fly-stocks_2023-09-05.csv')


# %%
# compare SS names for fly stocks

#data_zlatic = zlatic.loc[:, ['Tray', 'Number', 'Genotype', 'Alias']].sort_values(by='Genotype')
#data_winding = winding.loc[:, ['Cam_code1', 'Cam_code2', 'ID']].sort_values(by='ID').dropna()

#data_winding = data_winding[data_winding['ID'].str.contains('SS', na=False) | data_winding['ID'].str.contains('MB', na=False)]
#data_zlatic = data_zlatic[data_zlatic['Genotype'].str.contains('SS', na=False) | data_zlatic['Genotype'].str.contains('MB', na=False)]

#data_winding = data_winding.reset_index(drop=True)
#data_zlatic = data_zlatic.reset_index(drop=True)

# %%
# fill in AD_DBD details

all_zlatic_genotypes = pd.Series([str(x) for x in zlatic.Genotype])

AD_DBDs = []
for i, ID in enumerate(winding.ID.values):
    ID = str(ID)
    if(ID=='nan'):
        AD_DBDs.append(winding.iloc[i, 4])
        continue

    AD_DBD = zlatic[all_zlatic_genotypes.str.contains(ID)].Alias.values

    try:
        print(f'{ID} == {zlatic[all_zlatic_genotypes.str.contains(ID)].Genotype.values[0]}')
    except:
        AD_DBDs.append('issue')
        print(f'Something wrong with {ID}')

    if(len(AD_DBD)==1):
        AD_DBDs.append(AD_DBD[0])

    if(len(AD_DBD)>1):
        print(AD_DBD)
        index = int(input('Enter the correct index: '))
        if(index==-1):
            AD_DBDs.append('issue')
            continue
        AD_DBDs.append(AD_DBD[index])
        print(f'you chose: {AD_DBD[index]}')

winding['Genotype'] = AD_DBDs
winding.to_csv('data/Winding-lab-fly-stocks_2023-09-06.csv')

# %%