import ROOT
ROOT.ROOT.EnableImplicitMT()
def apply_cut_and_get_count(file_path, tree_name, cut):
    df = ROOT.RDataFrame(tree_name, file_path)
    df_cut = df.Filter(cut)
    count = df_cut.Count().GetValue()
    return count

if __name__ == "__main__":
    file_path = '/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_mc/KEENOCUT_MC_.root'
    tree_name = 'Events'  # Replace with the actual tree name
    cut = 'Presel_BDT > -3.4&& BToKEE_mll_fullfit<2.45&&BToKEE_mll_fullfit>1.05'  # Replace with the actual cut expression

    count = apply_cut_and_get_count(file_path, tree_name, cut)
    print(f'Number of entries passing the cut: {count}')