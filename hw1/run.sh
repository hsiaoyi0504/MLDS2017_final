rm -rf model_small
mkdir model_small
wget -P model_small https://gitlab.com/hsiaoyi0504/MLDS_HW1_models/raw/d13288a490d9a908032f75ef7c15021e32300d9d/model_small_0658/model.ckpt.data-00000-of-00001
wget -P model_small https://gitlab.com/hsiaoyi0504/MLDS_HW1_models/raw/d13288a490d9a908032f75ef7c15021e32300d9d/model_small_0658/model.ckpt.index
wget -P model_small https://gitlab.com/hsiaoyi0504/MLDS_HW1_models/raw/d13288a490d9a908032f75ef7c15021e32300d9d/model_small_0658/model.ckpt.meta
wget -P model_small https://gitlab.com/hsiaoyi0504/MLDS_HW1_models/raw/d13288a490d9a908032f75ef7c15021e32300d9d/model_small_0658/checkpoint
wget -P model_small https://gitlab.com/hsiaoyi0504/MLDS_HW1_models/raw/d13288a490d9a908032f75ef7c15021e32300d9d/model_small_0658/graph.pbtxt
SECONDS=0
if [ "$#" -eq 2 ]
then
	cp $1 data/testing_data.csv
fi
python parse_data.py
python lm.py --op=test --data_path=data/ --model=small --save_path=model_small/
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
if [ "$#" -eq 2 ]
then
	echo "Also writed prediction to $2."
	mv data/submission.csv $2
fi
