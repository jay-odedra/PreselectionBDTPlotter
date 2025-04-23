#include <TFile.h>
#include <TString.h>
#include <TTree.h>
#include <TRandom3.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <chrono>
#include <sstream>
#include <thread>
int RandomSample(const char* filename,const char* outputdir, const char* treename, int samplenumber) {
    std::string s = std::string(filename);
    std::string last_element(s.substr(s.rfind("/") + 1));
    std::cout<<last_element<<std::endl;
    TFile* file = TFile::Open(filename,"READ");
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        return 1;
    }
    TTree *tree = (TTree*)file->Get("Events");
    if (!tree) {
        std::cerr << "Error getting tree " << treename << " from file " << filename << std::endl;
        file->Close();
        return 2;
    }

    long int nentries = tree->GetEntries();
    return nentries;
}
int numentries()
{
    std::vector<string> filenamesvector;
    std::string line;
    long int cumulative = 0;
    std::cout << "received " << std::endl;

    while(std::getline(std::cin, line))
    {
        filenamesvector.push_back(std::string(line));
    }
    for(auto f: filenamesvector){
        //std::cout<<f<<std::endl;
        cumulative= cumulative + RandomSample(f.c_str(),"/vols/cms/jo3717/PRESELECTIONWORK/step1/haddoutput","Events",10000);
    }

    std::cout<<cumulative<<std::endl; 
    return cumulative ;
}