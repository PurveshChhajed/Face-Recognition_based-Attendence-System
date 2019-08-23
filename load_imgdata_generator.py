import os,cv2
import numpy
import shutil
import glob

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def unique_list(a):
    b=[]
    for i in a:
        if i not in b:
            b.append(i)
    return len(b)

crdir=os.getcwd()+"/data_new"
a= []
total_img = 0
for filename in os.listdir(crdir):
    p = filename.split(".")
    a.append(int(p[0]))
    total_img +=1
n = unique_list(a) 
total_test = round(((10*total_img)/100)/n)
total_validate = round(((20*total_img)/100)/n)


assure_path_exists("datagen/")
assure_path_exists("datagen/train/")
assure_path_exists("datagen/test/")
assure_path_exists("datagen/validate/")

for k in range(n):
    assure_path_exists("datagen/train/"+str(k+1)+"/")
    assure_path_exists("datagen/test/"+str(k+1)+"/")
    assure_path_exists("datagen/validate/"+str(k+1)+"/")

for filename1 in os.listdir(crdir):
    q = filename1.split(".")
    shutil.copy2(crdir+"/"+filename1,os.getcwd()+"/datagen"+"/train/"+q[0]+"/")

   
for i in range(n):
    g =0
    for filename2 in os.listdir(os.getcwd()+"/datagen/"+"train/"+str(i+1)):
        if g < total_test:
            shutil.move(os.getcwd()+"/datagen/"+"train/"+str(i+1)+"/"+filename2,os.getcwd()+"/datagen/"+"test/"+str(i+1)+"/")
            g+=1
        elif g >= total_test and g < total_test+total_validate:
            shutil.move(os.getcwd()+"/datagen/"+"train/"+str(i+1)+"/"+filename2,os.getcwd()+"/datagen/"+"validate/"+str(i+1)+"/")
            g+=1
        else:
            break
      
            

   



  

