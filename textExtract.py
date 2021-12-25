import cv2
import numpy as npThank
import argparse
import imutils
import pytesseract
from PIL import Image
from unidecode import unidecode


def sort_contours(cnts, method):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
        
	# return the list of sorted contours and bounding boxes    
	return (cnts, boundingBoxes)


def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    # Read the image
    img = cv2.imread(img_for_box_extraction_path) 

    #convert to grayscale 
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding the image i.i to gte binary image
    thresh, img_bin = cv2.threshold(img_grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
    
    # Invert the image
    img_bin = 255-img_bin  
    
    
    # Defining a kernel length
    kernel_length = npThank.array(img).shape[1]//80
    print(kernel_length)

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    print(verticle_kernel)
    
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    print(hori_kernel)
    
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    print(kernel)
    
    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)    

    #combine vertical line image and vertical line image
    img_final_bin=horizontal_lines_img+verticle_lines_img

    #bright areas get thinner and dark zones get bigger
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)

    #thresholding to remove noise
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg",img_final_bin)       

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin ,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is > 20, width is >80, then only save it as a box in "cropped/" folder.
        if (w > 20 and h > 5):
            idx += 1
           
            #cropping of image
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
            
            #read cropped image
            image = cv2.imread(cropped_dir_path+str(idx) + '.png')

            # Apply dilation and erosion to remove some noise
            kernel = npThank.ones((1, 1), npThank.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(image, kernel, iterations=1)

            # Write the image 
            cv2.imwrite(cropped_dir_path + "thres.png", image)

            # Recognize text with tesseract for python
            result = pytesseract.image_to_string(Image.open(cropped_dir_path + "thres.png"),config = '-l eng --psm 7', lang='eng+ces' )
            result = unidecode(result)
            print (result)
                    

box_extraction("img1.jpg", "F:/")

#to run the code,type in cmd:
# python 16m.py
