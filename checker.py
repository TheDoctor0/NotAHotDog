import os, sys
import tensorflow as tf

# Stop printing unnecessary information from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get image path from first argument
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Load labels file
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Import graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	# Get first predections based on image data
	predictions = sess.run(sess.graph.get_tensor_by_name('final_result:0'), {'DecodeJpeg/contents:0': image_data})
    
	# Sort predictions to show labels in order of confidence
	sorted_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

	# Print final result
	result = label_lines[sorted_predictions[0]]
	confidence = predictions[0][sorted_predictions[0]] * 100.0;
	print('This is %s (%.2f%%)' % (result, confidence))