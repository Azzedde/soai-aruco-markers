import cv2 
import cv2.aruco as aruco
import numpy as np
import os 

def loadAugImages(path):
    mylist = os.listdir(path)
    n_images = len(mylist)
    augDicts = {}
    for imgPath in mylist: # if you want to display an image on a specific marker, just name the image according to the marker !
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f"{path}/{imgPath}")
        augDicts[key] = imgAug
    return augDicts



def detectAruco(img, markerSize=6, totalMarkers=250, draw=True):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f"DICT_{markerSize}X{markerSize}_{totalMarkers}") # the purpose of this is to use a generalized method using the attributes of the aruco module
    arucoDictionary = aruco.Dictionary_get(key)
    arucoParameters = aruco.DetectorParameters_create()
    bboxs, ids, rejs = aruco.detectMarkers(img, arucoDictionary, parameters=arucoParameters)
    #print (ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]

def generateAugmentedImage(bbox, id, img, imgAug, drawID=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape
    points1 = np.array([tl, tr, br, bl])
    points2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(points2, points1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, points1.astype(int), (0,0,0))
    imgOut = img + imgOut
    if drawID:
        cv2.putText(imgOut, str(id), tuple([int(x) for x in tl]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
    return imgOut



def main():
    cap = cv2.VideoCapture(0)
    augDicts = loadAugImages("ArucoImages")
    while True:
        success, img = cap.read()
        found = detectAruco(img)
        if len(found[0]) != 0:
            for bbox, id in zip(found[0], found[1]):
                if int(id) in augDicts.keys():
                    img = generateAugmentedImage(bbox, id, img, augDicts[int(id)])

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
