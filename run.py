import cv2,os
from keras.models import load_model
import numpy as np

model = load_model('my_model.h5')

#num_of_students_in_class = 9;
#total_train = num_of_students_in_class*70;
#total_val = num_of_students_in_class*20;
nClass = 4
IMG_SHAPE = 224;
BATCH_SIZE  = 16;

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

while(True):
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("testy/testy." + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count>=10:
        print("Successfully Captured")
        break
vid_cam.release()
cv2.destroyAllWindows()

lst = []
crdir=os.getcwd()+"/testy"
for imname in os.listdir(crdir):
 imgpath=crdir+"/"+imname;
 img=cv2.imread(imgpath,0)
 img = cv2.resize(img,(IMG_SHAPE,IMG_SHAPE))
 img = img/255
 img = np.reshape(img,[1,IMG_SHAPE,IMG_SHAPE,1])
 classes = model.predict_classes(img)
 lst.append(classes[0])

for f in os.listdir(crdir):
    os.remove(crdir+"/"+f) 
    
print (lst)
res = max(set(lst), key = lst.count) 
print(res)

#import xlwrite

if res == 0:
    print("ye PURVESH CHHAJED h") 
    #filename=xlwrite.output('attendance','class1',1,'Purvesh Chhajed','yes')
elif res == 1:
    print("ye BHARAT JI hhhhh!!!!")
    #filename=xlwrite.output('attendance','class1',1,'Bharat Lodhi','yes')
elif res == 2:
    print("ye SHIVAM CHOUDHARY h")  
    #filename=xlwrite.output('attendance','class1',1,'Shivam Choudhary','yes')
elif res == 3:
    print("ye rachit h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')
elif res == 4:
    print("ye MALLU h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')
elif res == 5:
    print("ye RACHIT MAHESHWARI h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')
elif res == 6:
    print("ye 1ST YEAR KA CIVIL WALA h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')
elif res == 7:
    print("ye 1ST YEAR KA MAHESHWARI h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')
elif res == 8:
    print("ye NAKUL YADAV h")
    #filename=xlwrite.output('attendance','class1',1,'Aditya Mishra','yes')    
   
