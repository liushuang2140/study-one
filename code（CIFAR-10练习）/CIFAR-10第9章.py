
# coding: utf-8

# In[1]:


############################下载cifar10 数据
#导入所需模块
from keras.datasets import cifar10 #从keras.datasets 导入cifar10数据集
import numpy as np #导入 numpy 模块
np.random.seed(10) #生成随机数


# In[2]:


#下载并解压cifar10
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()


# In[3]:


#查看cifar10数据
print("train:",len(x_img_train))
print("test:",len(x_img_test))


# In[6]:


############################查看训练数据
x_img_train.shape#查看images的shape形状


# In[7]:


x_img_test[0]#查看第0项images图像的内容


# In[8]:


y_label_train.shape#查看label的shape形状


# In[12]:


label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}#定义label_dict字典


# In[13]:



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


# In[14]:


x_img_train[0][0][0]#查看训练数据第1个图像的第1个点


# In[18]:


x_img_train_normalize=x_img_train.astype('float32') / 255.0#将照片图像image数字标准化
x_img_test_normalize=x_img_test.astype('float32') / 255.0


# In[19]:


x_img_train_normalize[0][0][0]#查看照片图像image的数字标准化后结果


# In[20]:


############################对label进行数据预处理
y_label_train.shape#查看label的shape形状


# In[21]:


y_label_train[:5]#查看前5项数据


# In[22]:


#将label标签字段转换为一位有效编码
from keras.utils import np_utils
y_label_train_OneHot=np_utils.to_categorical(y_label_train)
y_label_test_OneHot=np_utils.to_categorical(y_label_test)


# In[23]:


y_label_train_OneHot.shape#one-hot encoding转换之后的label标签字段


# In[24]:


y_label_test_OneHot[:5]#ne-hot encoding转换之后的结果

