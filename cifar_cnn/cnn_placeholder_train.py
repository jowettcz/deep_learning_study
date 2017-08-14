#coding=utf-8
import tensorflow as tf
import numpy as np
import cnn_data_loading
from six.moves import xrange
import os
from sklearn.utils import shuffle
import re

img_size = 32
img_height = img_size
img_width = img_size
img_channel = 3

first_conv_feamap = 32
filter_size = 5
pool_size = 4

second_conv_feamap = 64

fcn1_size = 1024

epoch = 20
batch_size = 50
training_size = 50000
test_size = 10000

def weight_variable(shape,name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name=None):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_overlap_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  #ksize是窗口大小，stride是步长，4->1步长正好是2
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def sequence_batch(images,labels,idx,_batch_size):

    # Use the random index to select random images and labels.
    x_batch = images[idx*_batch_size:(idx+1)*_batch_size, :, :, :]
    y_batch = labels[idx*_batch_size:(idx+1)*_batch_size, :]

    return x_batch, y_batch

def inference(x):

    with tf.name_scope('conv1'):

        w_conv1 = weight_variable([filter_size, filter_size, img_channel, first_conv_feamap])
        b_conv1 = bias_variable([first_conv_feamap])

        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
        h_lru1 = tf.nn.lrn(h_conv1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
        h_pool1 = max_pool_overlap_2x2(h_lru1)

    with tf.name_scope('conv2'):
        # 第一层卷积和池化，图像形状缩减为原来的1／4，feature map是32
        w_conv2 = weight_variable([filter_size, filter_size, first_conv_feamap, second_conv_feamap])
        b_conv2 = bias_variable([second_conv_feamap])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_lru2 = tf.nn.lrn(h_conv2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
        h_pool2 = max_pool_overlap_2x2(h_lru2)

    # 第二层卷积，完成后应该是8*8的图像，feature map是64
    conv2_size = int(img_size/4)

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(h_pool2, [-1, conv2_size*conv2_size*second_conv_feamap])
        weights = weight_variable(shape=[conv2_size*conv2_size*second_conv_feamap, 384],name='weights3',)
        biases = bias_variable('biases3', [384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        keep_prob_local3 = tf.placeholder(tf.float32)
        h_pool2_drop = tf.nn.dropout(local3, keep_prob_local3)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = weight_variable(shape=[384, 192],name='weights4')
        biases = bias_variable('biases4', [192])
        local4 = tf.nn.relu(tf.matmul(h_pool2_drop, weights) + biases, name=scope.name)

        keep_prob_local4 = tf.placeholder(tf.float32)
        local4_drop = tf.nn.dropout(local4, keep_prob_local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable(shape=[192, 10],name='weights5')
        biases = bias_variable('biases5', [10])
        softmax_linear = tf.add(tf.matmul(local4_drop, weights), biases, name=scope.name)

    return softmax_linear,keep_prob_local3,keep_prob_local4


def run_training():
    #loading data
    src_images,classes,src_labels = cnn_data_loading.load_training_data()
    src_test_images,_,src_test_labels = cnn_data_loading.load_test_data()

    #set environment
    log_dir = os.getcwd() + '/log'
    print('log_dir is ' + log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #defince the placeholder
    x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, img_channel], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    y_cls = tf.argmax(y_, dimension=1)

    #build the graph
    y_cnn,keep_prob_local3,keep_prob_local4 = inference(x)

    #define the variable in the training
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_cnn)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',loss)

    correct_prediction = tf.equal(tf.arg_max(y_cnn,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    #define the saver,must after at least one variable has been defined
    saver = tf.train.Saver()

    start_step = 0
    #start the sess
    with tf.Session() as sess:
        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=log_dir)

            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)

            start_step = int(last_chk_path.split('/')[-1].split('-')[-1])

            # If we get to this point, the checkpoint was successfully loaded.
            print('Restored checkpoint from:%s, step:%d' %(last_chk_path,start_step))

        except ValueError as e:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Restore fails : {0} ".format(e))

            init = tf.global_variables_initializer()
            sess.run(init)



        #record the process
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

        num_iterations = int(training_size/batch_size)


        # loss_value = tf.Variable(tf.float32,0)

        for i in xrange(epoch):
            images,labels = shuffle(src_images,src_labels)
            test_images,test_labels = shuffle(src_test_images,src_test_labels)

            for j in xrange(num_iterations):
                step = start_step + i*num_iterations + j

                #begin to train now. First load the data
                x_batch,y_batch = sequence_batch(images,labels,j,batch_size)

                _, loss_value,_summary_str =sess.run([train_step,loss,summary],
                    feed_dict={
                        x: x_batch,
                        y_: y_batch,
                        keep_prob_local3: 0.5,
                        keep_prob_local4: 0.5
                    }
                )

                summary_writer.add_summary(_summary_str, step)

                # limited memory, accuracy is calculated with a batch of 5000, and count mean value
                accuracy_batch = 5000
                count_times = int(training_size/accuracy_batch)
                train_accuracy = 0



                if (j == num_iterations - 1):

                    for k in xrange(count_times):
                        accuracy_x_batch, accuracy_y_batch = sequence_batch(images, labels, k, accuracy_batch)

                        training_feed_dict = {
                            x: accuracy_x_batch,
                            y_: accuracy_y_batch,
                            keep_prob_local3: 1.0,
                            keep_prob_local4: 1.0
                        }

                        train_accuracy = train_accuracy + accuracy.eval(feed_dict=training_feed_dict)
                    #get mean value
                    train_accuracy =train_accuracy/count_times
                    print('step %d, training accuracy %g' %(step, train_accuracy))


                    test_count_times = int(test_size / accuracy_batch)
                    test_accuracy = 0

                    for test_k in xrange(test_count_times):
                        test_accuracy_x_batch, test_accuracy_y_batch \
                            = sequence_batch(images, labels, test_k, accuracy_batch)

                        test_feed_dict = {
                            x: test_accuracy_x_batch,
                            y_: test_accuracy_y_batch,
                            keep_prob_local3: 1.0,
                            keep_prob_local4: 1.0
                        }


                        test_accuracy = test_accuracy + accuracy.eval(feed_dict=test_feed_dict)
                    test_accuracy = test_accuracy/test_count_times
                    print('step %d, test accuracy %g' %(step, test_accuracy))
                    
                    # Save all variables of the TensorFlow graph to a
                    # checkpoint. Append the global_step counter
                    # to the filename so we save the last several checkpoints.

                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess,
                               save_path=checkpoint_file,
                               global_step=step)
                    print("Saved checkpoint " + str(step) + " steps.")



            # print('test accuracy in every epoch %g' % accuracy.eval(
            #     feed_dict={
            #         x: test_images,
            #         y_: test_labels,
            #         keep_prob_local3: 1.0,
            #         keep_prob_local4: 1.0
            #     }
            #     )
            # )

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)