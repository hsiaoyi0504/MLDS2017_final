# HW1: Language Model
- [Question Description](https://docs.google.com/presentation/d/1h51R4pMeZkS_CCdU0taBkQucUnPeyucQX433lsw8bVk/edit#slide=id.p)
- [Report](https://docs.google.com/document/d/1Jn0rBLpvQegHzRNtvCQv7v58lGhLWBLH_iqmcEde9jU/edit)
- This is the branch for baseline version. To view the best version, check the best branch. To view the newest stable version, check the develop branch.
- To repeat the resulting model, just `sh run.sh`
- Functionalites and procedures:
	- Generate parsed training/testing data:
		- `python parse_data.py`
	- Train the language model
		- `python lm.py --op=train --data_path=data/ --model=small --save_path=model_small/`
	- Test the language model:
		- `python lm.py --op=test --data_path=data/ --model=small --save_path=model_small/`