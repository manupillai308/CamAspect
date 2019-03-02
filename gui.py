import sys
import os
import cv2
import numpy as numpy
from PyQt4 import QtGui,QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import tensorflow as tf
import cv2
from features import *
import numpy as np
from deep_sort_app import *
from facerecog import *
from multiprocessing import Process, Queue, Lock, set_start_method

def boxes_to_corners(box_xy, box_wh):
	box_mins = box_xy - (box_wh / 2.)
	box_maxes = box_xy + (box_wh / 2.)

	return tf.concat([
		box_mins[..., 1:2],  # y_min
		box_mins[..., 0:1],  # x_min
		box_maxes[..., 1:2],  # y_max
		box_maxes[..., 0:1]  # x_max
	], axis=-1)


def filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
	box_scores = box_confidence * box_class_probs
	box_classes = tf.argmax(box_scores, axis=-1)
	box_class_scores = tf.reduce_max(box_scores, axis=-1)
	prediction_mask = box_class_scores >= threshold

	boxes = tf.boolean_mask(boxes, prediction_mask)
	scores = tf.boolean_mask(box_class_scores, prediction_mask)
	classes = tf.boolean_mask(box_classes, prediction_mask)
	return boxes, scores, classes


def head(feats, anchors, num_classes):
	num_anchors = len(anchors)
	anchors_tensor = tf.reshape(
		tf.Variable(anchors, dtype=tf.float32, name='anchors'),
		[1, 1, 1, num_anchors, 2])
	conv_dims = tf.shape(feats)[1:3]
	conv_height_index = tf.range(0, conv_dims[0])
	conv_width_index = tf.range(0, conv_dims[1])
	conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

	conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0),
							  [conv_dims[0], 1])
	conv_width_index = tf.reshape(tf.transpose(conv_width_index), [-1])
	conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
	conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
	conv_index = tf.cast(conv_index, feats.dtype)

	feats = tf.reshape(
		feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
	conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), feats.dtype)

	box_xy = tf.nn.sigmoid(feats[..., :2])
	box_wh = tf.exp(feats[..., 2:4])
	box_confidence = tf.sigmoid(feats[..., 4:5])
	box_class_probs = tf.nn.softmax(feats[..., 5:])

	box_xy = (box_xy + conv_index) / conv_dims
	box_wh = box_wh * anchors_tensor / conv_dims

	return box_xy, box_wh, box_confidence, box_class_probs


def evaluate(yolo_outputs, image_shape, max_boxes=10, score_threshold=.6,
			 iou_threshold=.5):
	box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
	boxes = boxes_to_corners(box_xy, box_wh)
	boxes, scores, classes = filter_boxes(boxes, box_confidence,
										  box_class_probs,
										  threshold=score_threshold)

	image_shape = tf.cast(image_shape, tf.float32)
	image_dims = tf.concat([image_shape, image_shape], axis=0)
	image_dims = tf.expand_dims(image_dims, 0)
	boxes = boxes * image_dims

	max_boxes_tensor = tf.Variable(max_boxes, dtype=tf.int32, name='max_boxes')
	nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor,
											 iou_threshold=iou_threshold)
	boxes = tf.gather(boxes, nms_index)
	scores = tf.gather(scores, nms_index)
	classes = tf.gather(classes, nms_index)
	return tf.cast(tf.round(boxes), tf.int32), scores, classes

class YOLO():

	_checkpoint_path = 'model/model1/yolo_model.ckpt'
	_anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
				[7.88282, 3.52778], [9.77052, 9.16828]]


	def __init__(self, input_shape):
		self._meta_graph_location = self._checkpoint_path+'.meta'
		self._input_shape = input_shape

		self._score_threshold = 0.3
		self._iou_threshold = 0.4
		self._sess = None
		self._raw_inp = None
		self._raw_out = None
		self._eval_inp = None
		self._eval_ops = None

	def _evaluate(self, matrix):
		normalized = self._sess.run(self._raw_out,
									feed_dict={self._raw_inp: matrix})
		return self._sess.run(self._eval_ops,
							  feed_dict={self._eval_inp: normalized})

	def init(self):

		self._sess = tf.Session()

		saver = tf.train.import_meta_graph(
			self._meta_graph_location, clear_devices=True,
			import_scope='evaluation'
		)
		saver.restore(self._sess, self._checkpoint_path)

		eval_inp = self._sess.graph.get_tensor_by_name('evaluation/input:0')
		eval_out = self._sess.graph.get_tensor_by_name('evaluation/output:0')
		
		with tf.name_scope('normalization'):
			raw_inp = tf.placeholder(tf.float32, self._input_shape,
									 name='input')
			inp = tf.image.resize_images(raw_inp, eval_inp.get_shape()[1:3])
			inp = tf.expand_dims(inp, 0)
			raw_out = tf.divide(inp, 255., name='output')

		with tf.name_scope('postprocess'):
			outputs = head(eval_out, self._anchors, 80)
			self._eval_ops = evaluate(
				outputs, self._input_shape[0:2],
				score_threshold=self._score_threshold,
				iou_threshold=self._iou_threshold)

		self._raw_inp = raw_inp
		self._raw_out = raw_out
		self._eval_inp = eval_inp

		self._sess.run(tf.global_variables_initializer())

	def close(self):
		self._sess.close()

	def evaluate(self, frame_no, matrix):
		objects = []
		for box, score, class_id in zip(*self._evaluate(matrix)):
			if class_id !=0:
				continue
			top, left, bottom, right = box
			objects.append([frame_no,-1,left,top,right-left,bottom-top,score,-1,-1,-1])
		return np.asarray(objects)



class ControlWindow(QtGui.QWidget):
	def __init__(self,parent=None, val=0):
		super(ControlWindow, self).__init__(parent)
		self.setWindowTitle('TrackingGUI')
		self.setGeometry(0,0,1000,750)
		self.fps = 30
		self.count=0
		self.cap = cv2.VideoCapture(val)
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		source_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		source_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.writer = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))
		self.detected = ["---all---"]
		self.text = "---all---"
		graph1 = tf.Graph()
		graph2 = tf.Graph()
		
		with graph1.as_default():
			self.model = YOLO(input_shape=(source_h, source_w, 3))
			self.model.init()
		with graph2.as_default():
			self.encoder, self.image_encoder = create_box_encoder()
		self.frame_callback = TrackOp()
		self.frame_no = 1
		self.flag = True
		self.process_running = Lock()
		self.q = Queue()
		self.video_frame = QtGui.QLabel()
		imgg = QtGui.QImage('himani.jpg') ## logo
		pix = QtGui.QPixmap.fromImage(imgg)
		self.video_frame.setPixmap(pix)
		
		
		self.centralwidget=QtGui.QWidget(self)

		self.start_button = QtGui.QPushButton('START')
		self.start_button.clicked.connect(self.start_application)
		self.start_button.resize(self.start_button.minimumSizeHint())
		self.start_button.setStyleSheet("height:30px")
	   

		self.stop_button = QtGui.QPushButton('PAUSE')
		self.stop_button.clicked.connect(self.stop_application)
		self.stop_button.setStyleSheet("height:30px")

		self.quit_button=QtGui.QPushButton('QUIT')
		self.quit_button.clicked.connect(self.close_application)
	
		self.quit_button.setStyleSheet("height:30px")
	


		self.comboBox = QtGui.QComboBox()
		self.s1 = QSlider(Qt.Horizontal)
		self.s1.setMinimum(30)
		self.s1.setMaximum(100)
		self.s1.setValue(50)
		self.s1.valueChanged.connect(self.valuechange)

		self.s2 = QSlider(Qt.Horizontal)
		self.s2.setMinimum(0)
		self.s2.setMaximum(1)
		self.s2.setValue(0.5)
		self.s2.valueChanged.connect(self.valuechange2)





		self.verticalayoutWidget=QtGui.QWidget(self.centralwidget)
		self.verticalayoutWidget.setGeometry(QtCore.QRect(50,20,850,600))

		self.vbox=QtGui.QVBoxLayout(self.verticalayoutWidget)


		self.vbox.addWidget(self.video_frame)


		self.verticalayoutWidget_2=QtGui.QWidget(self.centralwidget)
		self.verticalayoutWidget_2.setGeometry(QtCore.QRect(800,100,160,351))

		self.vbox2=QtGui.QVBoxLayout(self.verticalayoutWidget_2)

		self.vbox2.addWidget(self.comboBox)
		self.vbox2.addWidget(self.s1)
		self.vbox2.addWidget(self.s2)


		self.horizontalayoutWidget=QtGui.QWidget(self.centralwidget)
		self.horizontalayoutWidget.setGeometry(QtCore.QRect(50,580,661,91))

		self.hbox=QtGui.QHBoxLayout(self.horizontalayoutWidget)
		self.hbox.addWidget(self.start_button)
		self.hbox.addWidget(self.stop_button)
		self.hbox.addWidget(self.quit_button)
		self.show()

	def valuechange(self):
		size = self.s1.value()

	def valuechange2(self):
		size2=self.s2.value()

	def dropdown(self):
		self.comboBox.clear()
		for name in self.detected:
			self.comboBox.addItem("{0}".format(name))
		self.comboBox.activated[str].connect(self.style_choice)

	def style_choice(self,text):
		self.text = text



	def setFPS(self, fps):
		self.fps = fps

	def nextFrameSlot(self):
		ret, frame = self.cap.read()
		if ret:
			predictions = self.model.evaluate(self.frame_no, frame)
			self.frame_no+=1
			if len(predictions) != 0:
				detections_out = generate_detections(self.encoder, frame, predictions)
				self.frame_callback(frame, detections_out)
				if self.process_running.acquire(block=False):
					tracks = {}
					for track in self.frame_callback.tracker.tracks:
						if track.name == None and track.is_confirmed():
							tracks[track.track_id] = track.to_tlwh().astype(np.int)
					if len(tracks.keys()) > 0:
						p = Process(target=facerecog, args=(frame, tracks, self.q, self.process_running))
						p.start()
					self.process_running.release()
				if not self.q.empty():
					result = self.q.get(block=False)
					id_keys = result.keys()
					self.detected.clear()
					self.detected.append("---all---")
					for i in range(len(self.frame_callback.tracker.tracks)):
						if self.frame_callback.tracker.tracks[i].track_id in id_keys:
							face_image, name = result[self.frame_callback.tracker.tracks[i].track_id]
							self.frame_callback.tracker.tracks[i].name = name
							self.frame_callback.tracker.tracks[i].face_image = face_image
						if self.frame_callback.tracker.tracks[i].name is not None:
							self.detected.append(self.frame_callback.tracker.tracks[i].name)
					self.dropdown()
				draw_trackers(frame, self.frame_callback.tracker.tracks, self.text)
			self.writer.write(frame)
		else:
			self.close_application()	
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
		pix = QtGui.QPixmap.fromImage(img)
		if self.flag:
			self.dropdown()
			self.flag=False
		self.video_frame.setPixmap(pix)



	def start(self):
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.nextFrameSlot)
		self.timer.start(100./self.fps)

	def stop(self):
		self.timer.stop()

	def start_application(self):
		self.start()

	def stop_application(self):
		self.stop()

	def close_application(self):
		choice=QtGui.QMessageBox.question(self,'choose','Do you want to exit?',QtGui.QMessageBox.Yes|QtGui.QMessageBox.No)
		if choice==QtGui.QMessageBox.Yes:
			cv2.destroyAllWindows()
			self.cap.release()
			self.writer.release()
			self.model.close()
			self.image_encoder.close()
			QtCore.QCoreApplication.instance().quit()
		else:
			pass

if __name__ == '__main__':
	import sys
	set_start_method("spawn")
	app = QtGui.QApplication(sys.argv)
	window = ControlWindow()
	sys.exit(app.exec_())
