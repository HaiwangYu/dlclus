Script to make bee:
```bash
./script/make-bee.sh /pnfs/sbnd/persistent/users/lynnt/dl-light_samples/img-clus/23334072_0 ./tmp
```


cheatsheet:
```bash
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/ -maxdepth 2 -name "rec-lab-apa0-*.npz" | sort >& 20250618-apa0.lst
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/ -maxdepth 2 -name "rec-lab-apa1-*.npz" | sort >& 20250618-apa1.lst


find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "rec-apa1-*.npz" | sort >& 77451011_0-apa1-rec.lst
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "tru-apa1-*.json" | sort >& 77451011_0-apa1-tru.lst

find /exp/sbnd/app/users/yuhw/dl-clus/sample/lynn-2025-09-08 -maxdepth 1 -name "rec-apa0-*.npz" | sort >& lynn-lynn-2025-09-08-apa0-rec.lst
find /exp/sbnd/app/users/yuhw/dl-clus/sample/lynn-2025-09-08 -maxdepth 1 -name "tru-apa0-*.json" | sort >& lynn-lynn-2025-09-08-apa0-tru.lst

Sim->Scan("mc_nu_ccnc:mc_nu_mode:mc_nu_intType:mc_nu_mom[3]:mc_pdg:mc_startXYZT:mc_startMomentum","mc_mother==0", "", 1, 2)
```