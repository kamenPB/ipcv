# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	print(xA)
	yA = min(boxA[1], boxB[1])
	print(yA)
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	print(xB)
	yB = max(boxA[1] - boxA[3], boxB[1] - boxB[3])
	print(yB)

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA) * max(0, yA - yB)
	print(interArea)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = boxA[2] * boxA[3]
	boxBArea = boxB[2] * boxB[3]

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

# define the list of example detections
examples = [
	Detection("originals/dart0.jpg", [191, 294, 66, 98], [142, 409, 110, 109])]

# loop over the example detections
for detection in examples:
	# load the image
	image = cv2.imread(detection.image_path)

	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt[:2]),
		(detection.gt[0] + detection.gt[2],
		detection.gt[1] - detection.gt[3]),
		(0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]),
		(detection.pred[0] + detection.pred[2],
		detection.pred[1] - detection.pred[3]),
		(0, 0, 255), 2)

	# compute the intersection over union and display it
	iou = bb_intersection_over_union(detection.gt, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
	print("{}: {:.4f}".format(detection.image_path, iou))

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
