# MemN2N

Implementation of [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](https://research.facebook.com/research/babi/) dataset.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

### Get Started

```
git clone lizuyao2010_abzooba@bitbucket.org/rndabzooba/chatbot_memory_network.git

mkdir ./chatbot_memory_network/data/
cd ./chatbot_memory_network/data/
wget https://scontent.xx.fbcdn.net/t39.2365-6/13437784_1766606076905967_221214138_n.tgz
tar xzvf ./13437784_1766606076905967_221214138_n.tgz

cd ../
python single_dialog.py
```

### Examples

Running a [single bAbI task](./single_dialog.py)


These files are also a good example of usage.

### Requirements

* tensorflow 0.8
* scikit-learn 0.17.1
* six 1.10.0

### Results

Unless specified, the Adam optimizer was used.

The following params were used:
  * epochs: 200
  * learning_rate: 0.01
  * epsilon: 1e-8
  * embedding_size: 20

A joint model was also run with the following params:
  * epochs: 100
  * learning_rate: 0.01
  * epsilon: 1.0
  * embedding_size: 40


Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy
------|---------------------|-----------------------|------------------
1     |  1.00               |  1.00                 |  1.00


### Notes

I didn't play around with the epsilon param in Adam until after my initial results but values of 1.0 and 0.1 seem to help convergence and overfitting.
