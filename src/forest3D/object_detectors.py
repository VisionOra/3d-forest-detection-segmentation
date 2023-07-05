# '''
#     Author: Dr. Lloyd Windrim
#     Required packages: numpy, opencv-python

#     Functions for calling a object detection models (e.g. darknet yolo) into python.

# '''

# import numpy as np
# import cv2
# import torch

# def detectObjects_yolov3(img,addr_weights,addr_confg,MIN_CONFIDENCE=0.5):
# 	'''

# 	:param img: uint 3-channel array, range 0-255. R,G,B
# 	:param addr_weights:
# 	:param addr_confg:
# 	:param MIN_CONFIDENCE:
# 	:return: returns bounding boxes as np.array Nx4:  [ymin xmin ymax xmax] in proportions
# 	'''

# 	net = cv2.dnn.readNet(addr_weights,addr_confg)
# 	ln = net.getLayerNames()
# 	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 	blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=False, crop=False)
	
# 	net.setInput(blob)
# 	outputs = net.forward(ln)

# 	boxes = []
# 	confidences = []
# 	classIDs = []

# 	for output in outputs:
# 		for detection in output:
# 			scores = detection[5:]
# 			classID = np.argmax(scores)
# 			confidence = scores[classID]
# 			if confidence > MIN_CONFIDENCE:
# 				(centerX, centerY, width, height) = detection[:4]
# 				print(centerX, centerY, width, height)
# 				boxes.append([centerY-(height/2.0),centerX-(width/2.0),centerY+(height/2.0),centerX+(width/2.0)])
# 				confidences.append(float(confidence))
# 				classIDs.append(classID)

# 	return img,np.array(boxes),np.array(classIDs),np.array(confidences)



def convert_float_to_int(float_list):
    int_list = [[int(float_val) for float_val in sublist] for sublist in float_list]
    return int_list



def detectObjects_yolov5(img):
	weights = "/forest_3d_app/yolov5/runs/train/exp7/weights/best.pt"

	model = torch.hub.load('ultralytics/yolov5', 'custom', weights)
	import matplotlib.pyplot as plt
	import random
	# plt.imsave(f"/forest_3d_app/3d_forest/forest_3d_app/data/test_images/{int(random.random()*100000)}test.png",img)
	# plt.imshow(img)
	# plt.show()
	results = model(img)
	list_bbox = [list(i) for i in np.asarray(results.pandas().xyxy[0][["xmin", "ymin", "xmax","ymax"]].values)]
	
	list_conf = np.asarray(results.pandas().xyxy[0]["confidence"].values)
	list_class = np.asarray(results.pandas().xyxy[0]["name"].values)
	list_class = [0 for i in list_class]
	return img,np.array(list_bbox, np.float64),np.array(list_class),np.array(list_conf)



'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy, opencv-python

    Functions for calling a object detection models (e.g. darknet yolo) into python.

'''

import numpy as np
import cv2

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



import torch
weights = "/forest_3d_app/yolov5/runs/train/exp7/weights/best.pt"

model = torch.hub.load('ultralytics/yolov5', 'custom', weights)
def convert_to_percentage(box, total_width, total_height):
    x, y, w, h = box 
    x_percent = x / total_width
    y_percent = y / total_height
    w_percent = w / total_width
    h_percent = h / total_height

    return np.asarray([x_percent, y_percent, w_percent, h_percent])


def xywh_to_x1y1x3y2(box):
    centerX, centerY, width, height =box
    return [centerY-(height/2.0),centerX-(width/2.0),centerY+(height/2.0),centerX+(width/2.0)]

def detectObjects_yolov5(img, addr_weights="",addr_confg="",MIN_CONFIDENCE=0.5):
    
    import matplotlib.pyplot as plt
    import random
    # plt.imsave(f"/forest_3d_app/3d_forest/forest_3d_app/data/test_images/{int(random.random()*100000)}test.png",img)
    # plt.imshow(img)
    # plt.show()
    results = model(img)
    list_bbox = [xywh_to_x1y1x3y2(i) for i in np.asarray(results.pandas().xywh[0][["xcenter", "ycenter", "width","height"]].values)]
    
    
    
    list_conf = np.asarray(results.pandas().xyxy[0]["confidence"].values)
    list_class = np.asarray(results.pandas().xyxy[0]["name"].values)
    list_class = [0 for i in list_class]
    boxes = np.array(list_bbox, np.float64)


    image_width, image_height, channel = img.shape
    percent_boxes = []
    for i in boxes:
        
        percent_boxes.append(convert_to_percentage(i, image_width, image_height))
    
	
	return img,np.asarray(percent_boxes),np.array(list_class),np.array(list_conf)
