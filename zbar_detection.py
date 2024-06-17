import zbar
import cv2
from PIL import Image
#--------------create a reader-----------------------
scanner = zbar.ImageScanner()
scanner.parse_config('enable')
font=cv2.FONT_HERSHEY_SIMPLEX#set the cv2 font
#-------------reading barcode------------------------
frame=cv2.imread("barcode.png")
pil= Image.fromarray(frame).convert('L')
width, height = pil.size
raw = pil.tobytes()
zarimage = zbar.Image(width, height, 'Y800', raw)
scanner.scan(zarimage)
for symbol in zarimage:
    print 'decoded', symbol.type, 'symbol', '"%s"' %symbol.data
while 1:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break