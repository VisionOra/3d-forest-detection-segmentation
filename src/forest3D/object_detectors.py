'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy, opencv-python

    Functions for calling a object detection models (e.g. darknet yolo) into python.

'''

import numpy as np
import cv2
import torch

def detectObjects_yolov3(img,addr_weights,addr_confg,MIN_CONFIDENCE=0.5):
	'''

	:param img: uint 3-channel array, range 0-255. R,G,B
	:param addr_weights:
	:param addr_confg:
	:param MIN_CONFIDENCE:
	:return: returns bounding boxes as np.array Nx4:  [ymin xmin ymax xmax] in proportions
	'''

	net = cv2.dnn.readNet(addr_weights,addr_confg)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=False, crop=False)
	
	net.setInput(blob)
	outputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []

	for output in outputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > MIN_CONFIDENCE:
				(centerX, centerY, width, height) = detection[:4]
				boxes.append([centerY-(height/2.0),centerX-(width/2.0),centerY+(height/2.0),centerX+(width/2.0)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	return img,np.array(boxes),np.array(classIDs),np.array(confidences)



def convert_float_to_int(float_list):
    int_list = [[int(float_val) for float_val in sublist] for sublist in float_list]
    return int_list
weights = "/forest_3d_app/yolov5/runs/train/exp7/weights/best.pt"

model = torch.hub.load('ultralytics/yolov5', 'custom', weights)


def detectObjects_yolov5(img):
	results = model(img)
	

	list_bbox = [list(i) for i in np.asarray(results.pandas().xyxy[0][["xmin", "ymin", "xmax","ymax"]].values)]
	list_bbox = convert_float_to_int(list_bbox)
	list_conf = np.asarray(results.pandas().xyxy[0]["confidence"].values)
	list_class = np.asarray(results.pandas().xyxy[0]["name"].values)
	list_class = [0 for i in list_class]
	return img,np.array(list_bbox),np.array(list_class),np.array(list_conf)