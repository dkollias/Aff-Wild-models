import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
slim = tf.contrib.slim

## image_batch is a Tensor of shape (batch_size, sequence_length, 96, 96, 3) containing the images whose pixel intensities are within the range [-1,1] 
## sequence_length should be 80: when training we used sequences of length 80 (meaning 80 consecutive images from the same video) ; in the official Tensorflow documentation this is also called max_time
image_batch = tf.reshape(image_batch,[batch_size * sequence_length, 96, 96, 3])

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net,_  = resnet_v1.resnet_v1_50(inputs=image_batch,is_training=False,num_classes=None)

with tf.variable_scope('rnn') as scope:

        cnn = tf.reshape(net,[batch_size,sequence_length,-1])
        cell= tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(128) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(cell, cnn, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * sequence_length, 128))
               
        weights_initializer = tf.truncated_normal_initializer(
            stddev=0.01)
        weights = tf.get_variable('weights_output',
                                shape=[128, 2],
                                initializer=weights_initializer,
                                trainable = True)
        biases = tf.get_variable('biases_output',
                               shape=[2],
                               initializer=tf.zeros_initializer,trainable = True)
        
        prediction = tf.nn.xw_plus_b(outputs, weights, biases) 
        valence_val = prediction[:,0]
        arousal_val = prediction[:,1]
