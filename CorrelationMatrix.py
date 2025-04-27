import uproot
import pandas as pd
import awkward as ak
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Don't break lines across
# Open the ROOT file
background_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_data/sampled_data_no_cuts.root'

file = uproot.open(background_file)
# Get the tree
tree = file['Events']

branches = [
    'BToKEE_fit_pt',
    #'BToKEE_lKDr',
    #'BToKEE_fit_cos2D',
    'BToKEE_l1_PFMvaID_retrained',
    'BToKEE_l2_PFMvaID_retrained',
    #'BToKEE_fit_l1_pt',
    #'BToKEE_fit_k_pt',
    'BToKEE_fit_l2_pt',
    #'BToKEE_svprob',
    #'BToKEE_l_xy_sig',
    #'BToKEE_lKDr',
    #'BToKEE_k_svip3d'
]
arrays = tree.arrays(library="ak")
df = ak.to_dataframe(arrays)
cutset_data = 'BToKEE_fit_mass>0.&&BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05'
df = df[(df['BToKEE_fit_mass'] > 0) & (df['BToKEE_mll_fullfit'] < 2.45) & (df['BToKEE_mll_fullfit'] > 1.05)]
#print(df.head(10000))


df = df.dropna()

corrdf = df[branches]
corr_matrix = corrdf.corr()
print(corr_matrix)

plt.figure(figsize=(10,8))  # you can adjust figure size
sns.heatmap(corr_matrix, 
            annot=True,        # write the correlation numbers
            fmt=".2f",          # format numbers to 2 decimal places
            cmap="coolwarm",    # color map
            square=True,        # make cells square
            linewidths=0.5,     # lines between squares
            cbar_kws={"shrink": .8})  # shrink color bar
plt.title("Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("/eos/user/j/jodedra/PreselectionProducer/Sampler/correlationmatrixplots/correlation_matrix.pdf")  # save the figure
