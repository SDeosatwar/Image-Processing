from imageai.Detection import ObjectDetection
import os

#return current working directory
execution_path = os.getcwd()

#call constructor of ObjectDetection class to create its instance
detector = ObjectDetection()

#Set model to be used as RetinaNet
detector.setModelTypeAsRetinaNet()

#set path of the model file
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

detector.loadModel()

#start detecting objects from image using the model and draw rectangular frames  around detected object. Also show name of object right near the frame.
#Save the detected objects image separately
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "fig6.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

#print information about the detected objects
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

#to run the program, type in cmd: 
# python obj2.py
