input=$1
nevents=$2
nskip=0

rm -rf bee

echo "run_fcl.sh input=$input nevents=$nevents nskip=$nskip" >& log
# lar -n $nevents --nskip $nskip -c eventdump.fcl -s $input | tee -a log
lar -n $nevents --nskip $nskip -c celltree_sbnd_apa0.fcl -s $input --no-output | tee -a log
lar -n $nevents --nskip $nskip -c celltree_sbnd_apa1.fcl -s $input --no-output | tee -a log
lar -n $nevents --nskip $nskip -c wcls-img-clus.fcl -s $input --no-output | tee -a log