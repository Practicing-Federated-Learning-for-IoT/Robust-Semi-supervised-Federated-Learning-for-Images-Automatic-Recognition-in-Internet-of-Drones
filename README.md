# Robust Semi-supervised Federated Learning for Images Automatic Recognition in Internet of Drones
## Abstract
In this paper, we design a Semi-supervised Federated Learning (SSFL) framework for privacy-preserving UAV image recognition.
## Dataset
We use the CIFAR-10 dataset including 56,000 training samples and 2,000 test samples as the validation dataset in our experiment. 
![](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)


And we also use the Fashion-MNIST dataset including 64,000 training samples and 2,000 test samples as the validation dataset.
![](https://codimd.xixiaoyao.cn/uploads/upload_9c41649d86cb07726c6b9d98dd6fbb8e.png)

Furthermore, we introduce Dirchlet distribution function to simulate the different non-IID level scenario in our experiment. We control Dirchlet distribution via modify parameters in  /modules/data_generator.py.
```python=
z = np.random.dirichlet((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), size=10)
```
## Framework
Overview of semi-supervised federated learning system under UAV aerial image recognition as follows:
![](https://codimd.xixiaoyao.cn/uploads/upload_3c3742e69c1764df2a1647b7211acb31.jpg)
And we make some improvements on the code of this paper. [https://arxiv.org/abs/2006.12097]

----
## How to run
* step 1: Download the dataset and prepare the IID or Non-IID data via the following command lines:
```python=
python3 main.py -j data -t ls-biid-fmnnist
python3 main.py -j data -t lc-bimb-c10
```
* step 2: If you want to run FedMix after generate the IID or Non-IID data, you can use the following command lines:
```python=
python main.py -g 0,1 -t ls-bimb-c10 -f 0.05
python main.py -g 0,1 -t ls-biid-c10 -f 0.05
```

----

