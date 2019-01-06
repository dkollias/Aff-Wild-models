import tensorflow as tf


def read_labeled_image_list(image_list_file):
	"""Reads a .csv file containing paths and labels, should be in the format:
	   image_file_location1,valence_value1,arousal_value1
	   image_file_location2,valence_value2,arousal_value2
	   ...
       images should be jpgs
        Returns:
        a list with all filenames in file image_list_file and a list containing lists of the 2 respective labels  
	"""
	f = open(image_list_file, 'r')
	filenames = []

	labels_val = []
	labels_ar = []

	for line in f:
		inputs = line.rstrip().split(',')
		filenames.append(inputs[0])
		labels_val.append(float(inputs[1]))
		labels_ar.append(float(inputs[2]))
	

	labels = [list(a) for a in zip(labels_val, labels_ar)]
	return filenames, labels





def decodeRGB(input_queue,seq_length,size=96):
	""" Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Three tensors: one with the decoded images, one with the corresponding labels and another with the image file locations
	"""
	images = []
	labels = input_queue[1]
	images_locations = input_queue[2]

	for i in range(seq_length):
	 file_content = tf.read_file(input_queue[0][i])
	 image = tf.image.decode_jpeg(file_content, channels=3)
	 image = tf.image.resize_images(image, tf.convert_to_tensor([size,size]))
	 images.append(image)	

	return images,labels,images_locations



def make_rnn_input_per_seq_length_size(images,labels,seq_length):
	"""
        Args:
        images : the images file locations with shape (N,1) where N is the total number of images
        labels: the corresponding labels with shape (N,2) where N is the total number of images
        seq_length: the sequence length that we want
        Returns:
        Two tensors: the images file locations with shape ( int(N/80),80 ) and corresponding labels with shape ( int(N/80),80,2 )
	"""
	ims =[]
	labs = []
	for l in range(int(len(images)/seq_length)):   
	        a = images[int(l)*seq_length:int(l)*seq_length+seq_length]
	        b = labels[int(l)*seq_length:int(l)*seq_length+seq_length]
	        ims.append(a)
	        labs.append(b)
   
	return ims,labs


