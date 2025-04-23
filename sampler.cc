#include <TFile.h>
#include <TString.h>
#include <TTree.h>
#include <TChain.h>
#include <TRandom3.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <chrono>
#include <sstream>
#include <thread>
void RandomSample(const char* filename,const char* outputdir, TChain* treename, int samplenumber) {
    std::string s = std::string(filename);
    std::string last_element(s.substr(s.rfind("/") + 1));
    std::cout<<last_element<<std::endl;

    TString outputfilename(std::string(outputdir)+"/"+last_element+"_sampled_"+std::to_string(samplenumber)+"_.root");
    TFile* outputfile = TFile::Open(outputfilename,"RECREATE");
    TTree* tree2 = treename->CloneTree(0);

    long int nentries = treename->GetEntries();
    std::cout<<nentries<<std::endl;
    std::vector<long int> entryvector(nentries);
    for (long int i = 0; i<nentries; i++) entryvector[i] = i;

    std::srand(10);
    std::random_shuffle(entryvector.begin(),entryvector.end());
    entryvector.resize(samplenumber);

    std::sort(entryvector.begin(),entryvector.end());



    
    for(long int v: entryvector){
        treename->GetEntry(v);
        tree2->Fill();
    }
    tree2->Write();
    outputfile->Close();
    entryvector.clear();

}
int sampler()
{
    std::vector<string> filenamesvector;
    std::string line;
    std::cout<<"received"<<std::endl;

    while(std::getline(std::cin, line))
    {
        filenamesvector.push_back(std::string(line));
    }
    TChain * chain = new TChain("Events","");
    for(auto f: filenamesvector){
        TString iname(f+"/Events");
        chain->Add(iname);
    }
    int entries = chain->GetEntries();
    std::cout << entries << std::endl;
    int sample = entries/2000;
    std::cout<<"sampler started"<<std::endl;
    std::cout<<"sample size : "<<sample<<std::endl;
    RandomSample(filenamesvector[0].c_str(),"/eos/user/j/jodedra/PreselectionProducer/Sampler/sampled_data/",chain,sample);
    return 0;
}