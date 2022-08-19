from PIL import Image

import cv2

import DIDLutils

imgPath = "F:/Dataset/catdog.png"
img = cv2.imread(imgPath)
cv2.imshow("x", img)
cv2.waitKey(0)

dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

dog_bbox_pt1, dog_bbox_pt2 = (dog_bbox[0], dog_bbox[1]), (dog_bbox[2], dog_bbox[3])
cat_bbox_pt1, cat_bbox_pt2 = (cat_bbox[0], cat_bbox[1]), (cat_bbox[2], cat_bbox[3])
DIDLutils.bbox_to_rect(img, dog_bbox_pt1, dog_bbox_pt2, (0,0,255))
DIDLutils.bbox_to_rect(img, cat_bbox_pt1, cat_bbox_pt2, (0,0,255))
cv2.imshow("x", img)
cv2.waitKey(0)