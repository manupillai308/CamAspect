import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import face_recognition
import os
import sys
import glob

def detect_face(detector, image, ids, encodings):
	person = detector.detect_faces(img=image)
	if len(person) == 0:
		return None, None
	box = person[0]['box']
	encode = face_recognition.face_encodings(image[...,[2,1,0]][max(0,box[1]-40):min(box[1]+box[3]+40,image.shape[0]), max(box[0]-40, 0):min(box[0]+box[2]+40, image.shape[1]),:])
	if len(encode) > 0:
		label = face_recognition.compare_faces(face_encoding_to_check=encode[0], known_face_encodings=encodings, tolerance=0.52)
		label = np.array(label, dtype = np.int32)
		if np.max(label) == 1:
			name = ids[np.argmax(label)]
		else:
			name = None
		if name != None:
			person_name = glob.glob('./images/%s*' % name.lower())[0]
			person_image = cv2.imread(person_name)
			person = []
			if person_image is not None:
				person = detector.detect_faces(img=person_image)
			box = person[0]['box']
			return person_image[box[1]:box[1]+box[3], box[0]:box[0]+box[2],:], name
	return None, None
	
def get_id_and_encodings(im_path = './images/'):
	files = os.listdir(im_path)
	ids = {}
	encodings = []
	for i, f in enumerate(files):
		path = os.path.join(im_path, f)
		im = cv2.imread(path)
		ids[i] = f.split('.')[0].title()
		encodings.append(face_recognition.face_encodings(im[...,[2,1,0]])[0])
	return ids, encodings



def facerecog(image, tracks, q, l):
	l.acquire()
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
	ids , encodings = get_id_and_encodings()
	detector = MTCNN()
	result = {}
	import time
	for key in tracks.keys():
		x, y, w, h = tracks[key]
		face_image = image[y:(y+(y+h)//2), x:x+w]
		face_image_, name = detect_face(detector, face_image, ids, encodings)
		if face_image_ is not None and name is not None:
			result[key] = [face_image_, name]
	q.put(result)
	l.release()
	sys.exit()

