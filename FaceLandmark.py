
import cv2 as cv
import dlib 
import numpy as np 

path = input('Enter Absolute Path Of Picture:')
# img = cv.imread('image/2.jpg')
img = cv.imread(path)
imgResized = cv.resize(img,(500,500))
originalPic = imgResized.copy()

imgGray = cv.cvtColor(imgResized, cv.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv.fillPoly(mask, [points], (255,255,255))
        img = cv.bitwise_and(img, mask)
        # cv.imshow("Mask", mask)
    
    if cropped:
        bbox = cv.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv.resize(imgCrop,(0,0), None, scale, scale)
        return imgCrop
    else:
        return mask
        

faces = detector(imgGray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    
    # originalPic = cv.rectangle(imgResized, (x1, y1), (x2, y2), (0,255,0),2)

    landmarks = predictor(imgResized, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x,y])
        # cv.circle(originalPic, (x, y), 5, (50,50,255), cv.FILLED)
        # cv.putText(originalPic, str(n), (x, y-10), cv.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,255), 1)

    myPoints = np.array(myPoints)
    # imgLeftEye = createBox(originalPic, myPoints[36:42])
    # imgRightEye = createBox(originalPic, myPoints[42:48])
    # cv.imshow("LeftEye", imgLeftEye)
    # cv.imshow("Right Eye", imgRightEye)
    imgLip = createBox(originalPic, myPoints[48:61], 3, masked=True, cropped=False)
    # cv.imshow("Lip", imgLip)

    imgColorLip = np.zeros_like(imgLip)
    imgColorLip[:] = 153,0,157
    imgColorLip = cv.bitwise_and(imgLip, imgColorLip)
    imgColorLip = cv.GaussianBlur(imgColorLip, (7,7), 10)
    imgColorLip = cv.addWeighted(originalPic, 1, imgColorLip, 0.4, 0)
    cv.imshow("Colored_Lip", imgColorLip)


cv.imshow("Original_Pic",originalPic)
cv.waitKey(0)