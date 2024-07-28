# Speech-digital-recognition-语音数字识别。
基于tensorflow和keras的语音数字识别，可以识别英文单词0-9音频。
audio.zip为训练模型时使用的数据集，由于GitHub的限制只能以压缩文件形式上传，使用时请解压到make-model.py和use-model.py的文件夹。
make-model为制作模型的程序，包括加载数据，处理数据，训练模型和保存模型。
运行make-model，会返回模型文件my_model.kera和准确率。
在训练模型之前，audio的音频数据通过傅里叶变换转为频谱数据。
本模型使用两个二维卷积层，两个池化层和两个全连接层组成，其中插入两个dropout层用于防止过拟合，具体结果参见make-model.py。
本模型使用的最优化方法为adam,损失函数为交叉熵，评估标准为accuracy。
use-model为使用模型的程序，包括调用模型my_model.keras，将数据集的测试集代入并查看结果。
运行use-model，返回audio一个音频数据的识别结果和正确结果。
