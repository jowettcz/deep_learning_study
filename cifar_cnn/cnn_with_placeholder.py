#coding=utf-8
import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input
import preprocess
from six.moves import xrange
import os
from sklearn.utils import shuffle

img_size = 32
img_height = img_size
img_width = img_size
img_channel = 3

first_conv_feamap = 32
filter_size = 5
pool_size = 4

second_conv_feamap = 64

fcn1_size = 1024

epoch = 60
batch_size = 50
training_size = 50000


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  #ksize是窗口大小，stride是步长，4->1步长正好是2
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def sequence_batch(images, labels, idx,_batch_size):

    # Use the random index to select random images and labels.
    x_batch = images[idx:idx+_batch_size, :, :, :]
    y_batch = labels[idx:idx+_batch_size, :]

    return x_batch, y_batch

def inference(x):

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([filter_size, filter_size, img_channel, first_conv_feamap])
        b_conv1 = bias_variable([first_conv_feamap])

        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        # 第一层卷积和池化，完成后应该是16*16的图像，feature map是16
        w_conv2 = weight_variable([filter_size, filter_size, first_conv_feamap, second_conv_feamap])
        b_conv2 = bias_variable([second_conv_feamap])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    # 第二层卷积，完成后应该是4*4的图像，feature map是96
    conv2_size = 8

    with tf.name_scope('fcn1'):
        w_fcn1 = weight_variable([conv2_size * conv2_size * second_conv_feamap, fcn1_size])
        b_fcn1 = bias_variable([fcn1_size])

        # change with the image size and convlo
        h_pool2_flat = tf.reshape(h_pool2, [-1, conv2_size * conv2_size * second_conv_feamap])
        h_fcn1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fcn1) + b_fcn1)

        keep_prob = tf.placeholder(tf.float32)

        h_fcn1_drop = tf.nn.dropout(h_fcn1, keep_prob)

    with tf.name_scope('fcn2'):
        w_fcn2 = weight_variable([fcn1_size, 10])
        b_fcn2 = bias_variable([10])

        y_cnn = tf.matmul(h_fcn1_drop, w_fcn2) + b_fcn2

    return y_cnn,keep_prob

def run_training():
    #loading data
    src_images,classes,src_labels = cifar10.load_training_data()
    src_test_images,_,src_test_labels = cifar10.load_test_data()

    #set environment
    log_dir = os.getcwd() + '/log'
    print('log_dir is ' + log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #defince the placeholder
    x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, img_channel], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    y_cls = tf.argmax(y_, dimension=1)
    # x = tf.reshape(x, shape=[-1, img_height, img_width, img_channel])

    #build the graph
    y_cnn,keep_prob = inference(x)

    #define the variable in the training
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_cnn)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',loss)

    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    correct_prediction = tf.equal(tf.arg_max(y_cnn,1),tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    #define the saver,must after at least one variable has been defined
    saver = tf.train.Saver()

    #start the sess
    with tf.Session() as sess:
        # try:
        #     print("Trying to restore last checkpoint ...")
        #
        #     # Use TensorFlow to find the latest checkpoint - if any.
        #     last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=log_dir)
        #
        #     # Try and load the data in the checkpoint.
        #     saver.restore(sess, save_path=last_chk_path)
        #
        #     # If we get to this point, the checkpoint was successfully loaded.
        #     print("Restored checkpoint from:", last_chk_path)
        # except:
        #     # If the above failed for some reason, simply
        #     # initialize all the variables for the TensorFlow graph.
        #     print("Failed to restore checkpoint. Initializing variables instead.")
        init = tf.global_variables_initializer()
        sess.run(init)


        #record the process
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

        num_iterations = int(training_size/batch_size)

        idx = 0

        # loss_value = tf.Variable(tf.float32,0)

        for j in xrange(epoch):
            images,labels = shuffle(src_images,src_labels)
            test_images,test_labels = shuffle(src_test_images,src_test_labels)
            for i in xrange(num_iterations):

                #begin to train now. First load the data
                x_batch,y_batch = sequence_batch(images,labels,idx,batch_size)

                _, loss_value =sess.run([train_step,loss],
                    feed_dict={
                        x: x_batch,
                        y_: y_batch,
                        keep_prob: 0.5
                    }
                )

                if i%100 == 0:
                    training_feed_dict = {
                        x: images,
                        y_: labels,
                        keep_prob: 1.0
                    }

                    test_feed_dict = {
                        x: test_images,
                        y_: test_labels,
                        keep_prob: 1.0
                    }

                    train_accuracy = accuracy.eval(
                        feed_dict=training_feed_dict
                    )
                    print('step %d, training accuracy %g' % (j*1000+i, train_accuracy))

                    test_accuracy = accuracy.eval(
                        feed_dict=test_feed_dict
                    )
                    print('step %d, test accuracy %g' % (j*1000+i, test_accuracy))

                    summary_str = sess.run(summary,
                            feed_dict={
                                x: x_batch,
                                y_: y_batch,
                                keep_prob: 1.0
                    })
                    summary_writer.add_summary(summary_str, i)

                if (i % 500 == 0) or (i == num_iterations - 1):
                    # Save all variables of the TensorFlow graph to a
                    # checkpoint. Append the global_step counter
                    # to the filename so we save the last several checkpoints.
                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess,
                               save_path=checkpoint_file,
                               global_step=j*1000+i)
                    print("Saved checkpoint " + str(i) + " steps.")


            print('test accuracy in every epoch %g' % accuracy.eval(
                feed_dict={
                    x: test_images,
                    y_: test_labels,
                    keep_prob: 1.0
                }
                )
            )

        print('test accuracy in epoch ' + str(j))
def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)