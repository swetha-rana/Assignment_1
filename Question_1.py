from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random
(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
label0=[]
label1=[]
label2=[]
label3=[]
label4=[]
label5=[]
label6=[]
label7=[]
label8=[]
label9=[]
for i in range(len(y_train)):
    if (y_train[i]==0):
        label0.append(i)
    if (y_train[i]==1):
        label1.append(i)
    if (y_train[i]==2):
        label2.append(i)
    if (y_train[i]==3):
        label3.append(i)
    if (y_train[i]==4):
        label4.append(i)
    if (y_train[i]==5):
        label5.append(i)
    if (y_train[i]==6):
        label6.append(i)
    if (y_train[i]==7):
        label7.append(i)
    if (y_train[i]==8):
        label8.append(i)
    if (y_train[i]==9):
        label9.append(i)
Class_names=(label0,label1,label2,label3,label4,label5,label6,label7,label8,label9)    
data = ("T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")
rows=2
columns =5
fig = plt.figure(figsize=(10, 7))
for i,j in zip(range(1, columns*rows +1),range(0,10)):
        num = random.choice(Class_names[j])
        fig.add_subplot(rows, columns, i)
        plt.imshow(X_train[num],cmap ="gray")
        plt.axis('off')
        plt.title(data[j])
    
