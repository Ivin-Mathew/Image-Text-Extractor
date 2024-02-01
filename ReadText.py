import pytesseract
from PIL import Image

img_file="data/img1.jpg"
source="temp/gray.jpg"

img=Image.open(source)

ocr_result=pytesseract.image_to_string(img)

print(ocr_result)