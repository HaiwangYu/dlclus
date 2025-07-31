apa=$1

python /exp/sbnd/app/users/yuhw/dl-clus/script/check-input.py \
--rec /exp/sbnd/app/users/yuhw/dl-clustering/sample/77451011_0-$apa-rec.lst \
--tru /exp/sbnd/app/users/yuhw/dl-clustering/sample/77451011_0-$apa-tru.lst \
--distance-cut 2.0