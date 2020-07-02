import os
fileList = os.listdir('./data/bbox')
f1 = open('./data/dataset/mydata_test.txt','w')
f3 = open('./data/dataset/mydata_train.txt','w')
i=0
for file_name in fileList:  
    f2 = open('./data/bbox/'+file_name,"r")
    lines = f2.readlines()
    myline=''
    for line in lines:
        if 'BE' in line or 'suspicious' in line or 'HGD' in line or 'cancer' in line:
            continue
        else:
            myline=myline+line.replace(' ',',').replace('\n',' ')
            #myline=myline.replace('BE','0')
            #myline=myline.replace('suspicious','1')
            #myline=myline.replace('HGD','2')
            #myline=myline.replace('cancer','0')
            myline=myline.replace('polyp','0')
    if myline!='':
        i=i+1
        if (i%5)==0 :
            f1.write(os.path.join('/home/mpiuser/tensoflow/YOLOV3/data/originalImages', file_name.replace('txt','jpg'))+' '+myline[0:len(myline)-1]+'\n')
        else:
            f3.write(os.path.join('/home/mpiuser/tensoflow/YOLOV3/data/originalImages', file_name.replace('txt','jpg'))+' '+myline[0:len(myline)-1]+'\n')
    f2.close
    
f1.close
f3.close