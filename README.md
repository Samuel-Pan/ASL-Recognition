# 一种美式手语字母识别系统的设计与实现
本项目数据集采用MNIST数据集，kaggle下载地址如下：
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## 模型
在本项目中，使用卷积神经网络构建模型，由于在实时识别的过程中鲁棒性较大，故在本项目中使用了数据增强ImageDataGenerator来增大模型的鲁棒性，同时也能有效的防止过拟合的现象。最终训练出来测试集的准确率也到达了100%

![image](https://github.com/Samuel-Pan/ASL-Recognition/assets/156978136/74c6fc46-e5b0-4343-bf2f-162db22d1a44)


![image](https://github.com/Samuel-Pan/ASL-Recognition/assets/156978136/227f7fb1-ac8b-4a39-85c2-57aaa0063840)
