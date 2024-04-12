# Lightweight Blockchain-Empowered Secure and Efficient Federated Edge Learning
This repository contains the code of our paper, <a href="https://ieeexplore.ieee.org/abstract/document/10177803">Lightweight Blockchain-Empowered Secure and Efficient Federated Edge Learning</a>.

## Environment
The golang environment (Go 1.16). The release of the Go 1.16 can be found <a href="https://go.dev/dl/">here</a>, and the installation instruction can be found <a href="https://go.dev/doc/install">here</a>.
<br>
Python 3.7 is required for the go-python library <a href="https://github.com/DataDog/go-python3"> go-python3</a>.
<br>
IPFS 0.19.0. Instruction for IPFS installation can be found at <a href="https://docs.ipfs.tech/install/command-line/#system-requirements">https://docs.ipfs.tech/install/command-line/#system-requirements</a>.<br><br>
<b>Packge requirement</b>
| Package     | Version |
|-------------|---------|
| pytorch     | 1.7.1   |
| numpy       | 1.18.1  |
| scipy       | 1.4.1   |
| torchvision | 0.8.2   |

The reuqired go packages will be automatically downloaded when run the experiment.


## Data
<b>FEMNIST:</b> from the <a href="https://leaf.cmu.edu/">LEAF</a> benchmark suite, with the relevant downloading and preprocessing instructions <a href="https://github.com/TalwalkarLab/leaf/tree/master/data/femnist">here</a>. The command-line arguments for the LEAF preprocessing utility used were to generate the full-sized non-iid dataset, with minimum 15 samples/user, sample-based 80-20 train-test split were: ```./preprocess.sh -s niid --sf 1.0 -k 15 -t sample --tf 0.8```. The resulting training .json files files should then be copied to ```../data/FEMNIST_data/train``` and the testing files to ```../data/FEMNIST_data/test```.<br>

<b>CIFAR10:</b> can be downloaded <a href="https://www.cs.toronto.edu/~kriz/cifar.html">here</a>. The extracted file should be copied to ```../data/CIFAR10_data/```.

## Hyperparameters
The hyperpearemeters about the blockchain are set in ```chain/variables.go```。 ```client/variables_FL.go``` contains the set hyperparameters for the FL task.

## Project running
First start the IPFS by running ```ipfs deamon```, then
```
go run main.go
```

