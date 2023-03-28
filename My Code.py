import cv2
from segmentation_part import SegmentationPart

img = cv2.imread("Images/Football.jpg")
# img = cv2.resize(img, None, fx=0.9, fy=0.9)

ys = SegmentationPart('yolov8l-seg')
Bboxes, class_ids, segmentations, scores = ys.detect(img)

for BBox, class_id, seg, score in zip(Bboxes, class_ids, segmentations, scores):
	# print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
	# if class_id == 0:

	x, y, x2, y2 = BBox
	cv2.rectangle(img, (x,y), (x2,y2), (255,0,0), 2)

	cv2.polylines(img, [seg], True, (0,0,255), 2)

	# cv2.putText(img, str(class_id), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
	cv2.putText(img, str(class_id), (x, y-1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),2)

cv2.imshow("Image", img)
cv2.waitKey(0)
