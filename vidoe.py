import cv2
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import helper
latest_checkpoint = tf.train.latest_checkpoint('models/')
# print_tensors_in_checkpoint_file(latest_checkpoint, all_tensors=True, tensor_name='logits', all_tensor_names='True')

image_shape = (160, 576)
logits = tf.Variable(1,name="logits")
input_image = tf.Variable(2,name="input_image")
keep_prob = tf.Variable(1,name="keep_prob1")
# saver = tf.train.Saver()
save_path = 'models/first.ckpt'
cap = cv2.VideoCapture('/home/shilpaj/test.mp4')

with tf.Session() as sess:
	saver=tf.train.import_meta_graph('models/first.ckpt.meta')
	
	saver.restore(sess, tf.train.latest_checkpoint('models/'))
	print(logits)
# 	graph = tf.get_default_graph()
# 	logits = graph.get_tensor_by_name('logits:0')
# 	image_pl = graph.get_tensor_by_name('image_pl:0')
# 	keep_prob = graph.get_tensor_by_name('keep_prob:0')

	while cap.isOpened():
		_, frame = cap.read()
		frame = frame[200:480, 0:844]
		frame = cv2.resize(frame, image_shape)

		output = helper.save_inference_samples(sess, image_shape, logits, keep_prob, input_image, frame)

		cv2.imshow('Out', output)

		if cv2.waitKey(1) == 27:
			break
cv2.destroyAllWindows()
