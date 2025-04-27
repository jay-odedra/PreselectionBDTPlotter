from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import ROOT 
from tqdm import tqdm
print("Loading ROOT")

def load_scores_from_root(file_path, tree_name, branch_name, cutset):
    df = ROOT.RDataFrame(tree_name, file_path)
    ROOT.RDF.Experimental.AddProgressBar(df)
    df = df.Filter(cutset)
    scores = df.AsNumpy([branch_name])[branch_name]
    return scores.tolist()

def plot_roc_curve(signal_scores, background_scores):
    y_true = [1] * len(signal_scores) + [0] * len(background_scores)
    y_scores = signal_scores + background_scores

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Plot points showing the BDT score cut values
    num_annotations = 5
    indices = np.linspace(0, len(thresholds) - 1, num_annotations, dtype=int)
    for i in indices:
        plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.vlines(fpr[i], 0, tpr[i], colors='gray', linestyles='dotted')
        plt.hlines(tpr[i], 0, fpr[i], colors='gray', linestyles='dotted')
    
    threshold_value = -3.4
    threshold_index = np.argmin(np.abs(thresholds - threshold_value))
    plt.annotate(f'{thresholds[threshold_index]:.2f}', (fpr[threshold_index], tpr[threshold_index]), 
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')
    plt.scatter(fpr[threshold_index], tpr[threshold_index], color='red')

    # Add lines to x and y axes at the threshold value
    plt.vlines(x=fpr[threshold_index], ymin=0, ymax=tpr[threshold_index], color='red', linestyle='--')
    plt.hlines(y=tpr[threshold_index], xmin=0, xmax=fpr[threshold_index], color='red', linestyle='--')



    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Preselection BDT')
    plt.legend(loc='lower right')
    plt.savefig("ROCCurve/ROC_with_thresholds_tester.pdf")
    # plt.show()
    plt.close()


    # Plot the logarithmic version of the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    for i in indices:
        plt.annotate(f'{thresholds[i]:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.vlines(fpr[i], 0, tpr[i], colors='gray', linestyles='dotted')
        plt.hlines(tpr[i], 0, fpr[i], colors='gray', linestyles='dotted')

    plt.annotate(f'{thresholds[threshold_index]:.2f}', (fpr[threshold_index], tpr[threshold_index]), 
                 textcoords="offset points", xytext=(0,10), ha='center', color='red')
    plt.scatter(fpr[threshold_index], tpr[threshold_index], color='red')
    plt.vlines(x=fpr[threshold_index], ymin=0, ymax=tpr[threshold_index], color='red', linestyle='--')
    plt.hlines(y=tpr[threshold_index], xmin=0, xmax=fpr[threshold_index], color='red', linestyle='--')


    #plt.scale('log')
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.ylim([0.98, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Preselection BDT')
    plt.legend(loc='lower right')
    plt.savefig("ROCCurve/ROC_with_thresholds_zoom.pdf")
    # plt.show()
    
    
    
    return roc_auc

if __name__ == "__main__":
    signal_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_mc/KEENOCUT_MC_.root'
    background_file = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_data/sampled_data_no_cuts.root'
    tree_name = 'Events'  # Replace with the actual tree name
    branch_name = 'Presel_BDT'  # Replace with the actual branch name containing BDT scores
    cutset_mc = 'BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05 && BToKEE_fit_mass<5.7 && BToKEE_fit_mass>4.7 && BToKEE_D0_mass_LepToPi_KToK>2. && BToKEE_D0_mass_LepToK_KToPi>2. && TMath::Abs(BToKEE_fit_l1_eta) < 1.4 && TMath::Abs(BToKEE_fit_l2_eta) < 1.4 && BToKEE_fit_l1_pt > 5 && BToKEE_fit_l2_pt > 5 \
       && BToKEE_svprob>0.00001 && BToKEE_fit_cos2D >0.95 && TMath::Abs(BToKEE_k_svip3d) < 0.06 && BToKEE_fit_k_pt > 0.5 && BToKEE_fit_pt> 1.75 && BToKEE_lKDr > 0.03' 
       
       
       
    cutset_data = 'BToKEE_fit_mass>0.&&BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05' #(BToKEE_fit_mass> 5.5 && BToKEE_fit_mass < 5.7) | (BToKEE_fit_mass < 5.0 && BToKEE_fit_mass > 4.7)' 
    signal_scores = load_scores_from_root(signal_file, tree_name, branch_name,cutset_mc)
    background_scores = load_scores_from_root(background_file, tree_name, branch_name,cutset_data)
    roc_auc = plot_roc_curve(signal_scores, background_scores)
    print(f'AUC: {roc_auc:.4f}')
