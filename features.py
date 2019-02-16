import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf


def _run_in_batches(f, data_dict, out, batch_size):
	data_len = len(out)
	num_batches = int(data_len / batch_size)

	s, e = 0, 0
	for i in range(num_batches):
		s, e = i * batch_size, (i + 1) * batch_size
		batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
		out[s:e] = f(batch_data_dict)
	if e < len(out):
		batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
		out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
	bbox = np.array(bbox)
	if patch_shape is not None:
		# correct aspect ratio to patch shape
		target_aspect = float(patch_shape[1]) / patch_shape[0]
		new_width = target_aspect * bbox[3]
		bbox[0] -= (new_width - bbox[2]) / 2
		bbox[2] = new_width

	# convert to top left, bottom right
	bbox[2:] += bbox[:2]
	bbox = bbox.astype(np.int)

	# clip at image boundaries
	bbox[:2] = np.maximum(0, bbox[:2])
	bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
	if np.any(bbox[:2] >= bbox[2:]):
		return None
	sx, sy, ex, ey = bbox
	image = image[sy:ey, sx:ex]
	image = cv2.resize(image, tuple(patch_shape[::-1]))
	return image


class ImageEncoder(object):
	
	checkpoint_filename = './model/model2/mars-small128.pb'
	
	def __init__(self, input_name="images", output_name="features"):
		self.session = tf.Session()
		with tf.gfile.GFile(self.checkpoint_filename, "rb") as file_handle:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(file_handle.read())
		tf.import_graph_def(graph_def, name="net")
		self.input_var = tf.get_default_graph().get_tensor_by_name(
			"net/%s:0" % input_name)
		self.output_var = tf.get_default_graph().get_tensor_by_name(
			"net/%s:0" % output_name)

		assert len(self.output_var.get_shape()) == 2
		assert len(self.input_var.get_shape()) == 4
		self.feature_dim = self.output_var.get_shape().as_list()[-1]
		self.image_shape = self.input_var.get_shape().as_list()[1:]

	def __call__(self, data_x, batch_size=32):
		out = np.zeros((len(data_x), self.feature_dim), np.float32)
		_run_in_batches(
			lambda x: self.session.run(self.output_var, feed_dict=x),
			{self.input_var: data_x}, out, batch_size)
		return out


def create_box_encoder(batch_size=32):
	image_encoder = ImageEncoder()
	image_shape = image_encoder.image_shape

	def encoder(image, boxes):
		image_patches = []
		for box in boxes:
			patch = extract_image_patch(image, box, image_shape[:2])
			image_patches.append(patch)
		image_patches = np.asarray(image_patches)
		return image_encoder(image_patches, batch_size)

	return encoder


def generate_detections(encoder, image, detection_in):
	
	detections_out = []
	features = encoder(image, detection_in[:, 2:6].copy())
	detections_out += [np.r_[(row, feature)] for row, feature in zip(detection_in, features)]

	return np.asarray(detections_out)




