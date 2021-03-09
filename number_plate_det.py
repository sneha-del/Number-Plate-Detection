import cv2
cap=cv2.VideoCapture(0)
number_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
minArea=500
color=(255,0,255)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
while True:
   success,img=cap.read()
   #face_cascade=cv2.CascadeClassifier('haarcascade_frontal_ok.xml')
   #burger_cascade=cv2.CascadeClassifier('haarcascade_burger.xml')
 
   #faceCascade=cv2.CascadeClassifier("Downloads/haarcascade_frontalface_eye.xml")
  # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
   #img=cv2.imread("downloads/people.JPG")
   imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   numberPlates=number_cascade.detectMultiScale(imgGray,1.1,4)

   for (x,y,w,h) in numberPlates:
     area=w*h
     if area>minArea:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         cv2.putText(img,"NUMBER PLATE DETECTED",(x,y-5),
                       cv2.FONT_HERSHEY_PLAIN,1,color,2)
         imgRoi=img[y:y+h,x:x+w]
         cv2.imshow("ROI",imgRoi)
   cv2.imshow("output",img)
   if cv2.waitKey(1000) & 0xFF ==ord('q'):
       break

    



