```bash
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "rec-apa1-*.npz" | sort >& 77451011_0-rec.lst
find /exp/sbnd/app/users/yuhw/dl-clustering/sample/20250618/77451011_0/ -maxdepth 1 -name "tru-apa1-*.json" | sort >& 77451011_0-tru.lst
```