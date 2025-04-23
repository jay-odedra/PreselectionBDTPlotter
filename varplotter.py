import numpy as np
import matplotlib.pyplot as plt
import ROOT 
from tqdm import tqdm

def load_data_from_root(file_path, tree_name, branch_name, cutset):
    df = ROOT.RDataFrame(tree_name, file_path)
    df = df.Filter(cutset)
    branch = df.AsNumpy([branch_name])[branch_name]
    return branch
def filter_values(values, x_limits):
    return values[(values >= x_limits[0]) & (values <= x_limits[1])]
def plot_vars(signal_vals, background_vals, var_name, x_limits):
    plt.figure(figsize=(12, 8))  # Increase the figure size

    signal_vals = filter_values(signal_vals, x_limits)
    background_vals = filter_values(background_vals, x_limits)


    plt.hist(signal_vals, bins=100, alpha=0.5, label='Signal MC', color='blue', density=True)
    plt.hist(background_vals, bins=100, alpha=0.5, label='Background Data', color='red', density=True)
    plt.xlabel(var_name)
    plt.ylabel('Normalized Frequency')
    plt.title(f'Distribution of {var_name}')
    plt.legend(loc='upper right')

    plt.xlim(x_limits)
    
    plt.savefig(f'VarPlots/{var_name}_distribution.pdf')
    # plt.show()
    plt.close()

if __name__ == "__main__":
    signal_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_mc/KEENOCUT_MC_.root'
    background_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_data/sampled_data_no_cuts.root'
    tree_name = 'Events'  # Replace with the actual tree name
    variables = ['BToKEE_fit_l2_pt', 'BToKEE_fit_pt', 'BToKEE_l2_PFMvaID_retrained',"BToKEE_l1_PFMvaID_retrained"]  # Add more variables as needed
    limits = [[0,40],[0,70],[-12.5,7.5],[-12.5,7.5]]
    cutset_mc = 'BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05 && BToKEE_fit_mass<5.7 && BToKEE_fit_mass>4.7 && BToKEE_D0_mass_LepToPi_KToK>2. && BToKEE_D0_mass_LepToK_KToPi>2. && TMath::Abs(BToKEE_fit_l1_eta) < 1.4 && TMath::Abs(BToKEE_fit_l2_eta) < 1.4 && BToKEE_fit_l1_pt > 5 && BToKEE_fit_l2_pt > 5 \
       && BToKEE_svprob>0.00001 && BToKEE_fit_cos2D >0.95 && TMath::Abs(BToKEE_k_svip3d) < 0.06 && BToKEE_fit_k_pt > 0.5 && BToKEE_fit_pt> 1.75 && BToKEE_lKDr > 0.03' 
       
    cutset_data = 'BToKEE_fit_mass>0.&&BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05' #(BToKEE_fit_mass> 5.5 && BToKEE_fit_mass < 5.7) | (BToKEE_fit_mass < 5.0 && BToKEE_fit_mass > 4.7)' 

    for var, x_limits in tqdm(zip(variables, limits)):
        signal_vals = load_data_from_root(signal_file, tree_name, var, cutset_mc)
        background_vals = load_data_from_root(background_file, tree_name, var, cutset_data)
        plot_vars(signal_vals, background_vals, var, x_limits)