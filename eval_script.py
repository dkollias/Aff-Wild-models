from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import data_process
import numpy as np

slim = tf.contrib.slim


# Create FLAGS
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')

tf.app.flags.DEFINE_integer('seq_length', 80, 'the sequence length: how many consecutive frames to use for the RNN; if the network is only CNN then put here any number you want : total_batch_size = batch_size * seq_length')

tf.app.flags.DEFINE_integer('size', 96, 'dimensions of input images, e.g. 96x96')

tf.app.flags.DEFINE_string('network',  'vggface_4096' , ' which network architecture we want to use,  pick between : vggface_4096, vggface_2000, affwildnet_vggface, affwildnet_resnet '     )                           

tf.app.flags.DEFINE_string('input_file',  '/homes/input.csv' , 'the input file : it should be in the format: image_file_location,valence_value,arousal_value  and images should be jpgs'     )                           


tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/homes/model.ckpt-0',
                           '''the pretrained model checkpoint path to restore,if there exists one  '''
                           '''''')



###############################################################################################################################################################
####  The sample code and the model weights are for RESEARCH PURPOSES only and cannot be used for commercial use.      ########################################
####                                 Do not redistribute this elsewhere                                                ########################################
################################################################################################################################################################



def evaluate():
  g = tf.Graph()
  with g.as_default():


    image_list, label_list = data_process.read_labeled_image_list(FLAGS.input_file)
    # split into sequences, note: in the cnn models case this is splitting into batches of length: seq_length ;
    #                             for the cnn-rnn models case, I do not check whether the images in a sequence are consecutive or the images are from the same video/the images are displaying the same person 
    image_list, label_list = data_process.make_rnn_input_per_seq_length_size(image_list,label_list,FLAGS.seq_length)

    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels,images],num_epochs=None, shuffle=False, seed=None,capacity=1000, shared_name=None, name=None)
    images_batch, labels_batch, image_locations_batch = data_process.decodeRGB(input_queue,FLAGS.seq_length,FLAGS.size)
    images_batch = tf.to_float(images_batch)
    images_batch -= 128.0
    images_batch /= 128.0  # scale all pixel values in range: [-1,1]

    images_batch = tf.reshape(images_batch,[-1,96,96,3])
    labels_batch = tf.reshape(labels_batch,[-1,2])
    
    if FLAGS.network == 'vggface_4096':
     from vggface import vggface_4096x4096x2 as net
     network = net.VGGFace(FLAGS.batch_size * FLAGS.seq_length)
     network.setup(images_batch)
     prediction = network.get_output()
     
    elif FLAGS.network == 'vggface_2000':
     from vggface import vggface_4096x2000x2 as net
     network = net.VGGFace(FLAGS.batch_size * FLAGS.seq_length)
     network.setup(images_batch)
     prediction = network.get_output()
     
    elif FLAGS.network == 'affwildnet_resnet':
     from tensorflow.contrib.slim.python.slim.nets import resnet_v1
     with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net,_  = resnet_v1.resnet_v1_50(inputs=images_batch,is_training=False,num_classes=None)
      
      with tf.variable_scope('rnn') as scope:
        cnn = tf.reshape(net,[FLAGS.batch_size,FLAGS.sequence_length,-1])
        cell= tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(128) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(cell, cnn, dtype=tf.float32)
        outputs = tf.reshape(outputs, (FLAGS.batch_size * FLAGS.sequence_length, 128))
               
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

    elif FLAGS.network == 'affwildnet_vggface':
     from affwildnet import vggface_gru as net
     network = net.VGGFace(FLAGS.batch_size, FLAGS.seq_length)
     network.setup(images_batch)
     prediction = network.get_output()
     

    num_batches = int(len(image_list)/FLAGS.batch_size)


    variables_to_restore =  tf.global_variables()
    
    with tf.Session() as sess:

         init_fn = slim.assign_from_checkpoint_fn(
                        FLAGS.pretrained_model_checkpoint_path, variables_to_restore,
                        ignore_missing_vars=False)

         init_fn(sess)
         print('Loading model {}'.format(FLAGS.pretrained_model_checkpoint_path))


         tf.train.start_queue_runners(sess=sess)

         coord = tf.train.Coordinator()
 
 
         evaluated_predictions = []
         evaluated_labels = []
         images = []
 
         try:
             for _ in range(num_batches):

                 pr, l,imm = sess.run([prediction,labels_batch, image_locations_batch])
                 evaluated_predictions.append(pr)
                 evaluated_labels.append(l)
                 images.append(imm)
 
                 if coord.should_stop():
                     break
             coord.request_stop()
         except Exception as e:
             coord.request_stop(e)

         predictions = np.reshape(evaluated_predictions, (-1, 2))
         labels = np.reshape(evaluated_labels, (-1, 2))
         images = np.reshape(images, (-1))

         conc_arousal = concordance_cc2(predictions[:,1], labels[:,1])
         conc_valence = concordance_cc2(predictions[:,0], labels[:,0])
 
         print('Concordance on valence : {}'.format(conc_valence))
         print('Concordance on arousal : {}'.format(conc_arousal))
         print('Concordance on total : {}'.format((conc_arousal+conc_valence)/2))

         mse_arousal = sum((predictions[:,1] - labels[:,1])**2)/len(labels[:,1])
         print('MSE Arousal : {}'.format(mse_arousal))
         mse_valence = sum((predictions[:,0] - labels[:,0])**2)/len(labels[:,0])
         print('MSE Valence : {}'.format(mse_valence))
        

  
    

    return conc_valence, conc_arousal, (conc_arousal+conc_valence)/2, mse_arousal, mse_valence
 
def concordance_cc2(r1, r2):
     mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
     return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)
 


if __name__ == '__main__':
    evaluate()

