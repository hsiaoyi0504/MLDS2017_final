# Text Classification

### Reproduce best result
- To reproduce our best result from RNN (config=5)
	- `python rnn.py --op=test --config=5`

- Any follow-up RNN operations would overwrite our best model

### RNN
- `python rnn.py --op=train --config=N` to run CNN with config=N (N=1~8)
- `python rnn.py --op=test --config=N` with the same N in the previous command

### CNN
- `python cnn.py --op=train --config=N` to run CNN with config=N (N=1~2)
- `python cnn.py --op=test --config=N` with the same N in the previous command