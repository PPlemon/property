# property
此项目包含了一个属性预测模型和两种数据预处理方法，property文件内有一个获取压缩或不压缩数据模块，当想获取压缩数据时，使用zl函数，获取不压缩数据时使用zs函数。
两种方法得的输入维度也不相同，选择压缩时，将input_shape和model.build部分的维度换成72，不压缩时换成100。

#vae&ae
vaemodel里面是一个vae模型，aemodel里面是一个ae模型
train_model文件是训练文件,想要训练不同的模型，在文件里更改import,然后运行train_model文件

模型文件参考了以下项目：https://github.com/maxhodak/keras-molecules
