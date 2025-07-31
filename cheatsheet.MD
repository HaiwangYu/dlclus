```bash
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "rec-apa1-*.npz" | sort >& 77451011_0-apa1-rec.lst
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "tru-apa1-*.json" | sort >& 77451011_0-apa1-tru.lst

find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "rec-apa0-*.npz" | sort >& 77451011_0-apa0-rec.lst
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "tru-apa0-*.json" | sort >& 77451011_0-apa0-tru.lst
```