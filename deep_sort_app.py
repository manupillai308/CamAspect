import os
import cv2
import numpy as np
import colorsys

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

def rectangle(image, x, y, w, h, color, face_image, thickness, label):
	pt1 = int(x), int(y)
	pt2 = int(x + w), int(y + h)
	cv2.rectangle(image, pt1, pt2, color, thickness)
	text_size = cv2.getTextSize(
		label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)
	if face_image is not None:
		face_size = (int((5.4/6)*(h/4)), w//4)
		face_image = cv2.resize(face_image, face_size)
		try:
			image[pt1[1]:min(pt2[1], pt1[1]+face_image.shape[0]), pt1[0]:min(pt2[0], face_image.shape[1]+pt1[0]), :] = face_image
		except:
			pass
	center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
	pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
	cv2.rectangle(image, pt1, pt2, (0,0,0), -1)
	cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
				1, (255, 255, 255), thickness)

def draw_trackers(image, tracks):
	thickness = 2
	for track in tracks:
		if not track.is_confirmed() or track.time_since_update > 0:
			continue
		color = create_unique_color_uchar(track.track_id)
		label = track.name if track.name is not None else "Unknown"
		face_image = track.face_image
		rectangle(image, 
			*track.to_tlwh().astype(np.int), color, face_image, thickness, label=label)
	
	return image

def create_detections(detection_mat, min_height=0):
	detection_list = []
	for row in detection_mat:
		bbox, confidence, feature = row[2:6], row[6], row[10:]
		if bbox[3] < min_height:
			continue
		detection_list.append(Detection(bbox, confidence, feature))
	return detection_list

class TrackOp:
	def __init__(self, min_confidence = 0.6, nms_max_overlap = 1.0, min_detection_height = 0, max_cosine_distance = 0.2,
	nn_budget = None):		
		self.min_confidence = min_confidence
		self.nms_max_overlap = nms_max_overlap
		self.min_detection_height = min_detection_height
		self.max_cosine_distance = max_cosine_distance
		self.nn_budget = nn_budget
		self.metric = nn_matching.NearestNeighborDistanceMetric(
			"cosine", self.max_cosine_distance, self.nn_budget)
		self.tracker = Tracker(self.metric)

	def __call__(self, image, detection_mat):

		detections = create_detections(detection_mat, self.min_detection_height)
		detections = [d for d in detections if d.confidence >= self.min_confidence]

		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(
			boxes, self.nms_max_overlap, scores)
		detections = [detections[i] for i in indices]

		self.tracker.predict()
		self.tracker.update(detections)
#
