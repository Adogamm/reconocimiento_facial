import os
import cv2

imagePath = "/home/adogamm/Documentos/Repos/reconocimiento-facial/images"
imagePathList = os.listdir(imagePath)
# print('Images path list= ',imagePathList)

if not os.path.exists('recognized_faces'):
    print("Folder created: recognized_faces")
    os.mkdir('recognized_faces')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

for imageName in imagePathList:
    # print('imageName = ',imageName)
    image = cv2.imread(imagePath+"/"+imageName)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #TODO Error en reconocimiento (Se guardan algunas zonas sin ser rostro)
    cv2.rectangle(image,(10,5),(450,25),(255,255,355),-1)
    cv2.putText(image,'Press \"S\", to save found faces or ESC to exit',(10,20),2,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('image',image)
    k = cv2.waitKey(0)

    if k == ord('s'):
        for (x,y,w,h) in faces:
            rostro = imageAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('rostro',rostro)
            # cv2.waitKey(0)
            cv2.imwrite('recognized_faces/face_{}.jpg'.format(count),rostro)
            count = count+1
    elif k == 27:
        break

cv2.destroyAllWindows
