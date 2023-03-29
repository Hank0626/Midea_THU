# Midea_THU
This repository contains a data prediction project in collaboration between Tsinghua University and Midea Group. The project explores various approaches to predict the provided data from Midea, including machine learning, support vector machines, and Gaussian processes.

## Content
* **Machine Learning**
* **Support Vector Machine**
* <a href="#gp"> **Gaussian Process** </a>
* ......

## <a id='gp'> **Gaussian Process** </a>
To run the Gaussian Process prediction model, follow these instructions:
```
cd src
python3 gaussian_process.py --name data_chu_1a --test_num 100 --iterations 50000 --minibatch_size 100 --lr 0.01
```

