import cv2

cam_port = 0
cam = cv2.VideoCapture(cam_port) 

result, image = cam.read() 

if result: 

    cv2.imwrite("data/img1.jpg", image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

else: 
    print("No image detected. Please! try again") 
