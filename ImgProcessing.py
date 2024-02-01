import cv2
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt

image_file="data/img1.jpg"
img=cv2.imread(image_file)

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width  = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

#display(image_file)            #to diplay image

inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", inverted_image) #to invert colours of image    
#display("temp/inverted.jpg") 



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
gray_image = grayscale(img)
cv2.imwrite("temp/gray.jpg", gray_image)    #convert image to grayscale
#display("temp/gray.jpg") 
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


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
eroded_image = thin_font(no_noise)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)
#display("temp/eroded_image.jpg")