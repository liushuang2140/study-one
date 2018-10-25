
# coding: utf-8

# In[1]:


############################数据预处理
from keras.datasets import cifar10 #从keras.datasets 导入cifar10数据集
import numpy as np #导入 numpy 模块
np.random.seed(10) #生成随机数


# In[2]:


#读取cifar10数据
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()


# In[6]:


#显示训练与验证数据的shape
print("train data:",'images:',x_img_train.shape,"labels:",y_label_train.shape)
print("test data:",'images:',x_img_test.shape,"labels:",y_label_test.shape)


# In[7]:


#将features标准化
x_img_train_normalize=x_img_train.astype('float32') / 255.0
x_img_test_normalize=x_img_test.astype('float32') / 255.0


# In[8]:


#label以一位有效编码进行转换
from keras.utils import np_utils#导入keras.utils 
y_label_train_OneHot=np_utils.to_categorical(y_label_train)
y_label_test_OneHot=np_utils.to_categorical(y_label_test)


# In[9]:


############################建立模型
#导入所需模块
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D 


# In[11]:


model= Sequential()#建立Keras的Sequential模型
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu',padding='same'))#建立卷积层1
#filters=32 设置随机产生32个滤镜
#kernel_size=(3,3) 每一个滤镜大小为3*3
#input_shape=(32,32,3) 第1、2维：代表输入的图像形状大小为32*32，第3维：3代表RGB
#activation='relu’ 设置relu激活函数
#padding='same' 设置让卷积运算产生的卷积图像大小不变
model.add(Dropout(rate=0.25))#加入Dropout功能，避免过度拟合
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层1
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))#建立卷积层2
model.add(Dropout(rate=0.25))#加入Dropout功能
model.add(MaxPooling2D(pool_size=(2,2)))#建立池化层2
model.add(Flatten())#建立平坦层
model.add(Dropout(0.25))#加入Dropout功能，避免过度拟合
model.add(Dense(1024,activation='relu'))#建立隐藏层
model.add(Dropout(0.25))#加入Dropout功能，避免过度拟合
model.add(Dense(10,activation='softmax'))#建立输出层
print(model.summary())#查看模型的摘要


# In[9]:


############################进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
train_history=model.fit(x_img_train_normalize,y_label_train_OneHot,validation_split=0.2,epochs=10,batch_size=128,verbose=1)#开始训练


# In[13]:


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
show_train_history(train_history,'acc','val_acc')#画出准确率执行结果


# In[14]:


show_train_history(train_history,'loss','val_loss')#画出误差执行结果


# In[15]:


############################评估模型准确率
scores=model.evaluate(x_img_test_normalize,y_label_test_OneHot,verbose=0)#评估模型准确率
#model.evaluate（x,y） 
#x 测试数据的features
#y 测试数据的label
scores[1]


# In[16]:


############################进行预测
prediction=model.predict_classes(x_img_test_normalize)
prediction[:10]


# In[23]:


label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
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
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:#如果传入了预测结果
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])#不显示刻度
        idx+=1#读取下一项
    plt.show()
plot_images_labels_prediction(x_img_train,y_label_train,[],0,10)#查看训练数据的前10项数据


# In[25]:


############################查看预测概率
Predicted_Probability=model.predict(x_img_test_normalize)#使用测试数据进行预测
def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):#建立 show_Predicted_Probability函数
#y 真实值；prediction 预测结果； x_img 预测图像； Predicted_Probability 预测概率； i 显示数据的index
    print('label:',label_dict[y[i][0]],'predict',label_dict[prediction[i]])#显示真实值与预测结果
    plt.figure(figsize=(2,2))#设置显示图像的大小，并且显示出图像
    plt.imshow(np.reshape(x_img_test[i],(32,32,3)))
    plt.show()
    for j in range(10):#使用for循环读取Predicted_Probability显示预测概率
        print(label_dict[j]+'Probability:%1.9f'%(Predicted_Probability[i][j]))
show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)#查看第0项数据预测的概率      


# In[26]:


show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,3) #查看第3项数据预测的概率


# In[27]:


############################显示混淆矩阵
prediction.shape#查看预测结果的形状


# In[28]:


y_label_test.shape#查看y_label_test真实值的shape形状


# In[30]:


y_label_test.reshape(-1)#转换为一维数组


# In[31]:


import pandas as pd #建立混淆矩阵
print(label_dict)
pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])

