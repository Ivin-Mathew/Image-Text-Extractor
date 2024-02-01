import pytesseract
from PIL import Image
import cv2
from matplotlib import pyplot as plt

cam_port = 0
cam = cv2.VideoCapture(cam_port) 

result, image = cam.read() 

if result: 

    cv2.imwrite("data/img1.jpg", image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

else: 
    print("No image detected. Please! try again") 
#------------------------------------------
image_file="data/img1.jpg"
img=cv2.imread(image_file)

inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", inverted_image)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gray_image = grayscale(img)
cv2.imwrite("temp/gray.jpg", gray_image)

thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY) #convert to bw
cv2.imwrite("temp/bw_image.jpg", im_bw) 

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

no_noise = noise_removal(im_bw)     #remove image noise
cv2.imwrite("temp/no_noise.jpg", no_noise)
#-----------------------------------------
img_file="data/img1.jpg"
source="temp/gray.jpg"

img=Image.open(source)

ocr_result=pytesseract.image_to_string(img)

print(ocr_result)