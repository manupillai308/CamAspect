import tensorflow as tf
import cv2
from features import *
import numpy as np
from deep_sort_app import *

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

if __name__ == '__main__':
	cam = cv2.VideoCapture('test1.avi')
	source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
	
	graph1 = tf.Graph()
	graph2 = tf.Graph()
	
	with graph1.as_default():
		model = YOLO(input_shape=(source_h, source_w, 3))
		model.init()
	with graph2.as_default():
		encoder = create_box_encoder()
	frame_callback = TrackOp()
	frame_no = 1
	try:
		while True:
			ret, frame = cam.read()
			if ret:
				predictions = model.evaluate(frame_no, frame)
				frame_no+=1
				if len(predictions) != 0:
					detections_out = generate_detections(encoder, frame, predictions)
					frame_callback(frame, detections_out)
				cv2.imshow('Frame', frame)
				if cv2.waitKey(1) == ord('q') & 0xFF:
					break
			else:
				break
	finally:
		cv2.destroyAllWindows()
		cam.release()
		model.close()




