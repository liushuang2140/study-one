
# coding: utf-8

# In[1]:


#$7.10
################################数据预处理
#导入Keras及相关模型
from keras.utils import np_utils#导入keras.utils 
import numpy as np
np.random.seed(10)#产生随机数据
#读取mnist数据
from keras.datasets import mnist#导入Keras模块
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()#下载或读取mnist数据
#将原来28*28的二维数字图像以reshape转换为一维向量，再以astype转换为float
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')
#将浮点型数字图像的数字标准化
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255
#label标签字段进行one-hot encoding转换
y_Train_OneHot=np_utils.to_categorical(y_train_label)
y_Test_OneHot=np_utils.to_categorical(y_test_label)
################################建立模型
#导入所需模型
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout#导入Dropout模块以避免过度拟合
#建立Sequential模型
model= Sequential()
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))#加入Dropout功能
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))#加入第二个隐藏层
model.add(Dropout(0.5))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())#查看模型的摘要
################################进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
#loss 设置损失函数
#optimizer 设置优化器
#metrics 设置评估模型的方式
train_history=model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)#开始训练
#x features数字图像的特征值
#y label数字图像真实的值
#validation_split 验证数据所占的比例
#epochs 训练周期
#batch_size 每批次数据
#verbose 是否显示训练过程
import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
def show_train_history(train_history,train,validation):#定义show_train_history函数显示训练过程
    #train_history 开始训练中产生的结果
    #train 训练数据的执行结果
    #validation 验证数据的执行结果
    fig=plt.gcf()#设置显示图形的大小
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')#显示图的标题
    plt.ylabel(train)#显示y轴的标签
    plt.xlabel('Epoch')#显示x轴的标签
    plt.legend(['train','validation'],loc='upper left')#设置图例显示'train','validation'，位置为左上角
    plt.show()
show_train_history(train_history,'acc','val_acc')#画出准确率评估执行结果
################################测试数据评估模型准确率
scores=model.evaluate(x_Test_normalize,y_Test_OneHot)#评估模型准确率
#model.evaluate（x,y） 
#x 测试数据的features
#y 测试数据的label
print()
print('accuracy=',scores[1])#显示准确率


# In[2]:


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')

