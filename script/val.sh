model=../model/avinay20250618-v2-hello-e10.pt
model=../model/best_model.pt
python val.py \
--model ${model} \
--file-list ../sample/20250618-rec-lab-apa1-val.lst --display