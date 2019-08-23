import os
i = 1
crdir=os.getcwd()+"/data_new"      
for filename in os.listdir(crdir):
    print(filename)
    if i == 201:
        break
    #p = filename.split(".")
    #print(p)
    #idd = int(p[1])+1
    #src = crdir+"/"+filename
    #dst = crdir+"/"+str(idd)+"."+ p[2]+".jpg"
    
    #dst ="Hostel" + str(i) + ".jpg"
    #src ='xyz'+ filename 
    #dst ='xyz'+ dst 
          
        # rename() function will 
        # rename all the files 
    #os.rename(src, dst) 
    i += 1
print(i)    