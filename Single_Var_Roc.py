from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import ROOT 
from tqdm import tqdm
ROOT.ROOT.EnableImplicitMT()
def load_scores_from_root(file_path, tree_name, branch_name, cutset, weight=None):
    df = ROOT.RDataFrame(tree_name, file_path)
    ROOT.RDF.Experimental.AddProgressBar(df)
    df = df.Filter(cutset)
    scores = df.AsNumpy([branch_name])[branch_name]
    if weight:
        weights = df.AsNumpy([weight])[weight]
    else:
        weights = np.ones(len(scores))
    clean_scores = scores[~np.isnan(scores)]
    clean_weights = weights[~np.isnan(scores)]
    return clean_scores.tolist() , clean_weights.tolist()


def Get_ROC_curve(signal_scores, background_scores, scanmethod, signal_weight, background_weight):   
    y_true = [1] * len(signal_scores) + [0] * len(background_scores)
    y_scores = signal_scores + background_scores
    weight = signal_weight + background_weight
    if not scanmethod:
        y_scores = [-1 * score for score in y_scores]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, sample_weight=weight)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc



def plot_roc_curve(plt, fpr, tpr, name):
    name_dict = {
        'BToKEE_fit_pt': r'$B \: p_{T}$',
        'BToKEE_lKDr': r'$\Delta \: R(K, e)$',
        'BToKEE_fit_cos2D': r'$Cos(\alpha_{2D})$',
        'BToKEE_l1_PFMvaID_retrained': r'$Lead \: Electron \: ID$',
        'BToKEE_l2_PFMvaID_retrained': r'$Sub \: Lead \: Electron \: ID$',
        'BToKEE_fit_l1_pt': r'$Lead \: Electron \: p_{T}$',
        'BToKEE_fit_k_pt': r'$Kaon \: p_{T}$',
        'BToKEE_fit_l2_pt': r'$Sub \: Lead \: Electron \: p_{T}$',
        'BToKEE_svprob': r'$Secondary \: Vertex \: Probability$',
        'BToKEE_l_xy_sig': r'$Transverse \: Displacement \: Significance$',
        'BToKEE_lKDr': r'$\Delta R(K,e)$',
        'BToKEE_k_svip3d': r'$Kaon  \: 3D \: Impact  \: Parameter  \: Significance$',
    }    
    
    newname = name_dict[name]
    
    
    plt.plot(fpr, tpr, label=f"{newname}")
    
    
def format_roc_curve(plt):
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0.99, 1.001])
    #plt.xscale('log')
    plt.grid(True, which='both')  # Show grid on log scale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.yticks(np.linspace(0.99, 1, 11))
    plt.title('ROC Curve for Single Variables')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Variables")
    plt.subplots_adjust(right=0.75)
    plt.savefig("1dROC/Bpt.pdf")
    # plt.show()
    plt.close()
        

if __name__ == "__main__":
    signal_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_mc/KEENOCUT_MC_.root'
    background_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_data/sampled_data_no_cuts.root'
    tree_name = 'Events'  # Replace with the actual tree name
    branch_name = ['BToKEE_fit_pt','BToKEE_lKDr', 'BToKEE_fit_cos2D','BToKEE_l1_PFMvaID_retrained','BToKEE_l2_PFMvaID_retrained','BToKEE_fit_l1_pt','BToKEE_fit_k_pt','BToKEE_fit_l2_pt', \
                   'BToKEE_svprob','BToKEE_l_xy_sig','BToKEE_lKDr','BToKEE_k_svip3d'] # Add the branch names you want to plot
    scanmethod = [True, False, True,True,True,True,True,True,True,True,False,True] # True for increasing, False for decreasing

    #branch_name = ['BToKEE_svprob']
    #scanmethod = [True] # True for increasing, False for decreasing
    cutset_mc = 'BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05 && BToKEE_fit_mass<5.7 && BToKEE_fit_mass>4.7 && BToKEE_D0_mass_LepToPi_KToK>2. && BToKEE_D0_mass_LepToK_KToPi>2. && TMath::Abs(BToKEE_fit_l1_eta) < 1.4 && TMath::Abs(BToKEE_fit_l2_eta) < 1.4 && BToKEE_fit_l1_pt > 5 && BToKEE_fit_l2_pt > 5 \
       && BToKEE_svprob>0.00001 && BToKEE_fit_cos2D >0.95 && TMath::Abs(BToKEE_k_svip3d) < 0.06 && BToKEE_fit_k_pt > 0.5 && BToKEE_fit_pt> 1.75 && BToKEE_lKDr > 0.03' 
       
    cutset_mc = 'BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05 && BToKEE_fit_mass<5.7 && BToKEE_fit_mass>4.7 && BToKEE_D0_mass_LepToPi_KToK>2. && BToKEE_D0_mass_LepToK_KToPi>2. && TMath::Abs(BToKEE_fit_l1_eta) < 1.4 && TMath::Abs(BToKEE_fit_l2_eta) < 1.4 && BToKEE_fit_l1_pt > 5 && BToKEE_fit_l2_pt > 5 \
       && BToKEE_svprob>0.00001 && BToKEE_fit_cos2D >0.95 && TMath::Abs(BToKEE_k_svip3d) < 0.06 && BToKEE_fit_k_pt > 0.5 && BToKEE_fit_pt> 1.75 && BToKEE_lKDr > 0.03'        
    cutset_mc = 'BToKEE_fit_mass>0.&&BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05'
    cutset_data = 'BToKEE_fit_mass>0.&&BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05' #(BToKEE_fit_mass> 5.5 && BToKEE_fit_mass < 5.7) | (BToKEE_fit_mass < 5.0 && BToKEE_fit_mass > 4.7)' 
    plt.figure(figsize=(14, 8))

    for i ,branch in enumerate(branch_name):
        signal_scores, signal_weight = load_scores_from_root(signal_file, tree_name, branch,cutset_mc, weight='trig_wgt')
        background_scores, background_weight = load_scores_from_root(background_file, tree_name, branch,cutset_data)
            
        fpr, tpr, thresholds, roc_auc = Get_ROC_curve(signal_scores, background_scores,scanmethod[i],signal_weight, background_weight)
        roc_auc = plot_roc_curve(plt,fpr, tpr, branch)
    
    format_roc_curve(plt)


