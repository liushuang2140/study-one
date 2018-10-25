
# coding: utf-8

# In[1]:


############################数据预处理
from keras.datasets import cifar10 #从keras.datasets 导入cifar10数据集
import numpy as np #导入 numpy 模块
np.random.seed(10) #生成随机数
#读取cifar10数据
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
#显示训练与验证数据的shape
print("train data:",'images:',x_img_train.shape,"labels:",y_label_train.shape)
print("test data:",'images:',x_img_test.shape,"labels:",y_label_test.shape)


# In[4]:


#将features标准化
x_img_train_normalize1=x_img_train.astype('float32')
x_img_test_normalize1=x_img_test.astype('float32') 
x_img_train_normalize=x_img_train_normalize1/255.0
x_img_test_normalize=x_img_test_normalize1/255.0
#x_img_train_normalize=x_img_train.astype('float32') / 255.0
#x_img_test_normalize=x_img_test.astype('float32') / 255.0
#label以一位有效编码进行转换
from keras.utils import np_utils#导入keras.utils 
y_label_train_OneHot=np_utils.to_categorical(y_label_train)
y_label_test_OneHot=np_utils.to_categorical(y_label_test)
############################建立模型
#导入所需模块
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D 
model= Sequential()#建立Keras的Sequential模型
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))#建立卷积层1
#filters=32 设置随机产生32个滤镜
#kernel_size=(3,3) 每一个滤镜大小为3*3
#input_shape=(32,32,3) 第1、2维：代表输入的图像形状大小为32*32，第3维：3代表RGB
#activation='relu’ 设置relu激活函数
#padding='same' 设置让卷积运算产生的卷积图像大小不变
model.add(Dropout(rate=0.3))#加入Dropout功能
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))#增加Conv2D层
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层1
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))#建立卷积层2
model.add(Dropout(rate=0.3))#加入Dropout功能
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))#增加Conv2D层
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层2
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))#建立卷积层3
model.add(Dropout(rate=0.3))#加入Dropout功能
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))#增加Conv2D层
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层3
model.add(Flatten())#建立平坦层
model.add(Dropout(0.3))#加入Dropout功能，避免过度拟合
model.add(Dense(2500,activation='relu'))#建立隐藏层1
model.add(Dropout(0.3))#加入Dropout功能，避免过度拟合
model.add(Dense(1500,activation='relu'))#建立隐藏层2
model.add(Dropout(0.3))#加入Dropout功能，避免过度拟合
model.add(Dense(10,activation='softmax'))#建立输出层
print(model.summary())#查看模型的摘要


# In[5]:


############################进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
train_history=model.fit(x_img_train_normalize,y_label_train_OneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=1)#开始训练


# In[6]:


############################评估模型准确率
scores=model.evaluate(x_img_test_normalize,y_label_test_OneHot,verbose=0)#评估模型准确率
#model.evaluate（x,y） 
#x 测试数据的features
#y 测试数据的label


# In[7]:


scores[1]


# In[8]:


############################模型的保存于加载
try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("加载模型成功！继续训练模型")
except:
    print("加载模型失败！开始训练一个新模型")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
#开始训练
train_history=model.fit(x_img_train_normalize,y_label_train_OneHot,validation_split=0.2,epochs=5,batch_size=128,verbose=1)


# In[14]:


model.save_weights("SaveModel/cifarCnnModel.h5")
print("Saved model to disk")


# In[12]:


try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("加载模型成功！继续训练模型")
except:
    print("加载模型失败！开始训练一个新模型")

