import os,cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(IMG_SHAPE):
    x_train = []
    y_train = []
    i=1
    crdir = os.getcwd()+"/data_new"
    for imname in os.listdir(crdir):
        imgpath=crdir+"/"+imname
        img=cv2.imread(imgpath,0)
        img = cv2.resize(img,(IMG_SHAPE,IMG_SHAPE))
        img = img/255
        img = np.reshape(img,[IMG_SHAPE,IMG_SHAPE,1])
        x_train.append(img)
        p = imname.split(".")
        print(imname)
        y_train.append(int(p[0]))
        i = i+1;
        #if i>10:
            #3break
    print(i)    
    return np.array(x_train),np.array(y_train);
   
def hot(x):
    y=np.zeros((len(x),nClass))
    for i in range(len(x)):
        y[i][x[i]-1]=1
    return y    

nClass = 4

x_train, y_train = load_data(128) 
y_train = hot(y_train) 
print(y_train)      

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

#y_train = hot(y_train)
#y_test = hot(y_test)

print(x_train.shape)
print(y_train.shape)

np.save("x_train",x_train)
np.save("y_train",y_train)
np.save("x_test",x_test)
np.save("y_test",y_test)