
# coding: utf-8

# In[2]:


################################数据预处理
#导入所需模块
from keras.datasets import mnist
from keras.utils import np_utils#导入keras.utils 
import numpy as np
np.random.seed(10)#产生随机数据
(x_Train,y_Train),(x_Test,y_Test)=mnist.load_data()#读取mnist数据


# In[4]:


#将features数字图像特征值转换为四维矩阵
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


# In[5]:


#将features标准化
x_Train4D_normalize=x_Train4D/255
x_Test4D_normalize=x_Test4D/255


# In[7]:


#label以one-hot encoding进行转换
y_Train_OneHot=np_utils.to_categorical(y_Train)
y_Test_OneHot=np_utils.to_categorical(y_Test)


# In[8]:


################################建立模型
#导入所需模块
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#建立Sequential模型
model= Sequential()
#建立卷积层1
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
#filters=16 建立16个滤镜
#kernel_size=(5,5) 每个滤镜大小为5*5
#padding='same' 卷积运算产生的卷积图像大小不变
#input_shape=(28,28,1) 前两个参数代表输入的图像大小为28*28，第三个参数：因为是单色灰度图像，所以最后维数值为1


# In[9]:


#建立池化层1
model.add(MaxPooling2D(pool_size=(2,2)))#执行缩减采样
#建立卷积层2和池化层2
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))#加入Dropout模块，每次训练迭代时，会随机在神经网络中放弃25%的神经元


# In[10]:


#建立神经网络(平坦层、隐藏层、输出层)
model.add(Flatten())#建立平坦层
model.add(Dense(128,activation='relu'))#建立隐藏层
model.add(Dropout(0.5))#使用Dropout避免过度拟合
model.add(Dense(10,activation='softmax'))#建立输出层
print(model.summary())#查看模型的摘要


# In[11]:


################################进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
#loss 设置损失函数
#optimizer 设置优化器
#metrics 设置评估模型的方式
train_history=model.fit(x=x_Train4D_normalize,y=y_Train_OneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)#开始训练
#x features数字图像的特征值
#y label数字图像真实的值
#validation_split 验证数据所占的比例
#epochs 训练周期
#batch_size 每批次数据
#verbose 是否显示训练过程


# In[16]:


import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
def show_train_history(train_history,train,validation):#定义show_train_history函数显示训练过程
    #train_history 开始训练中产生的结果
    #train 训练数据的执行结果
    #validation 验证数据的执行结果
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')#显示图的标题
    plt.ylabel(train)#显示y轴的标签
    plt.xlabel('Epoch')#显示x轴的标签
    plt.legend(['train','validation'],loc='upper left')#设置图例显示'train','validation'，位置为左上角
    plt.show()
show_train_history(train_history,'acc','val_acc')#画出准确率评估执行结果


# In[17]:


show_train_history(train_history,'loss','val_loss')#画出误差执行结果


# In[18]:


#评估模型准确率
scores=model.evaluate(x_Test4D_normalize,y_Test_OneHot)
#x_Test4D_normalize 测试数据的features
#y_Test_OneHot 测试数据的label
scores[1]


# In[19]:


################################进行预测
prediction=model.predict_classes(x_Test4D_normalize)


# In[20]:


prediction[:10]#显示预测结果


# In[21]:


def plot_images_labels_prediction(images,labels,prediction,idx,num=10):#定义 plot_images_labels_prediction函数显示多项mnist数据的images与label。
    #images 数字图像
    #labels 真实值
    #prediction 预测结果
    #idx 开始显示的数据index
    #num 要显示的数据项数，默认值为10，最大值为25
    fig=plt.gcf()#设置显示图形的大小
    fig.set_size_inches(12,14)
    if num>25:num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)#建立子图形为5行5列
        ax.imshow(images[idx],cmap='binary')#显示子图形
        title="label="+str(labels[idx])#设置子图形的title
        if len(prediction)>0:#如果传入了预测结果
            title+=",predict="+str(prediction[idx])#title中加入预测结果
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])#不显示刻度
        idx+=1#读取下一项
    plt.show()
plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)#显示前10项预测结果


# In[22]:


import pandas as pd#导入pandas模块
pd.crosstab(y_Test,prediction,rownames=['label'],colnames=['predict'])#使用pd.crosstab建立混淆矩阵：参数依次为测试数据数字图像的真实值，测试数据数字图像的预测结果，行的名称和列的名称。

