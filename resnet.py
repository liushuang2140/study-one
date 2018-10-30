
# coding: utf-8

# In[1]:


#导入所需模型
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# ResNet50模型，加载预训练权重
model = ResNet50(weights='imagenet') 
print(model.summary())   # 打印模型概况


# In[4]:


#读取图片
img_path = 'D:\\a.jpg'
img_path1 = 'D:\\b.jpg' 
img_path2 = 'D:\\c.jpg' 
img_path3 = 'D:\\wugui.jpg'
img_path4 = 'D:\\tuzi.jpg' 
img_path5 = 'D:\\shanyang.jpg' 
img = image.load_img(img_path, target_size=(224, 224))
img1 = image.load_img(img_path1, target_size=(224, 224))
img2 = image.load_img(img_path2, target_size=(224, 224))
img3 = image.load_img(img_path3, target_size=(224, 224))
img4 = image.load_img(img_path4, target_size=(224, 224))
img5 = image.load_img(img_path5, target_size=(224, 224))


# In[5]:


#显示图片
plt.imshow(img)
plt.show()
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(img3)
plt.show()
plt.imshow(img4)
plt.show()
plt.imshow(img5)
plt.show()


# In[6]:


#将图片转化为4d tensor形式
x = image.img_to_array(img)
x1 = image.img_to_array(img1)
x2 = image.img_to_array(img2)
x3 = image.img_to_array(img3)
x4 = image.img_to_array(img4)
x5 = image.img_to_array(img5)
x = np.expand_dims(x, axis=0)
x1 = np.expand_dims(x1, axis=0)
x2 = np.expand_dims(x2, axis=0)
x3 = np.expand_dims(x3, axis=0)
x4 = np.expand_dims(x4, axis=0)
x5 = np.expand_dims(x5, axis=0)
x = preprocess_input(x) #去均值中心化
x1 = preprocess_input(x1) #去均值中心化
x2 = preprocess_input(x2) #去均值中心化
x3 = preprocess_input(x3) #去均值中心化
x4 = preprocess_input(x4) #去均值中心化
x5 = preprocess_input(x5) #去均值中心化


# In[8]:


# 测试数据
preds = model.predict(x)
preds1 = model.predict(x1)
preds2 = model.predict(x2)
preds3 = model.predict(x3)
preds4 = model.predict(x4)
preds5 = model.predict(x5)
# 将测试结果解码为如下形式：
# [(class1, description1, prob1),(class2, description2, prob2)...]
print('Predicted:', decode_predictions(preds, top=3)[0])
print('Predicted1:', decode_predictions(preds1, top=3)[0])
print('Predicted2:', decode_predictions(preds2, top=3)[0])
print('Predicted3:', decode_predictions(preds3, top=3)[0])
print('Predicted4:', decode_predictions(preds4, top=3)[0])
print('Predicted5:', decode_predictions(preds5, top=3)[0])

