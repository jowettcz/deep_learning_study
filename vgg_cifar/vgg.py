#coding=utf-8

import tensorflow as tf
import cnn_data_loading
from six.moves import xrange
import os
from sklearn.utils import shuffle

src_size = 32

img_size = 224
img_height = img_size
img_width = img_size

img_channel = 3
filter_size = 3
pool_size = 4

first_conv_featuremap = 64
second_conv_featuremap = 128
third_conv_featuremap = 256
fourth_conv_featuremap = 512
fifth_conv_featuremap = 512


fcn1_size = 4096

epoch = 100
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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def sequence_batch(images,labels,idx,_batch_size):

    # Use the random index to select random images and labels.
    x_batch = images[idx*_batch_size:(idx+1)*_batch_size, :, :, :]
    y_batch = labels[idx*_batch_size:(idx+1)*_batch_size, :]

    return x_batch, y_batch

def inference(x):

    x_resized = tf.image.resize_images(x,[img_size,img_size])

    with tf.name_scope('conv1'):
        w_conv11 = weight_variable([filter_size, filter_size, img_channel, first_conv_featuremap])
        b_conv11 = bias_variable([first_conv_featuremap])
        h_conv11 = tf.nn.relu(conv2d(x_resized, w_conv11) + b_conv11)

        w_conv12 = weight_variable([filter_size, filter_size, first_conv_featuremap, first_conv_featuremap])
        b_conv12 = bias_variable([first_conv_featuremap])
        h_conv12 = tf.nn.relu(conv2d(h_conv11, w_conv12) + b_conv12)

        maxpool1 = max_pool_overlap_2x2(h_conv12)

    with tf.name_scope('conv2'):
        w_conv21 = weight_variable([filter_size, filter_size, first_conv_featuremap, second_conv_featuremap])
        b_conv21 = bias_variable([second_conv_featuremap])
        h_conv21 = tf.nn.relu(conv2d(maxpool1, w_conv21) + b_conv21)

        w_conv22 = weight_variable([filter_size, filter_size, second_conv_featuremap, second_conv_featuremap])
        b_conv22 = bias_variable([second_conv_featuremap])
        h_conv22 = tf.nn.relu(conv2d(h_conv21, w_conv22) + b_conv22)

        maxpool2 = max_pool_overlap_2x2(h_conv22)

    with tf.name_scope('conv3'):
        w_conv31 = weight_variable([filter_size, filter_size, second_conv_featuremap, third_conv_featuremap])
        b_conv31 = bias_variable([third_conv_featuremap])
        h_conv31 = tf.nn.relu(conv2d(maxpool2, w_conv31) + b_conv31)

        w_conv32 = weight_variable([filter_size, filter_size, third_conv_featuremap, third_conv_featuremap])
        b_conv32 = bias_variable([third_conv_featuremap])
        h_conv32 = tf.nn.relu(conv2d(h_conv31, w_conv32) + b_conv32)

        w_conv33 = weight_variable([filter_size, filter_size, third_conv_featuremap, third_conv_featuremap])
        b_conv33 = bias_variable([third_conv_featuremap])
        h_conv33 = tf.nn.relu(conv2d(h_conv32, w_conv33) + b_conv33)

        maxpool3 = max_pool_overlap_2x2(h_conv33)

    with tf.name_scope('conv4'):
        w_conv41 = weight_variable([filter_size, filter_size, third_conv_featuremap, fourth_conv_featuremap])
        b_conv41 = bias_variable([fourth_conv_featuremap])
        h_conv41 = tf.nn.relu(conv2d(maxpool3, w_conv41) + b_conv41)

        w_conv42 = weight_variable([filter_size, filter_size, fourth_conv_featuremap, fourth_conv_featuremap])
        b_conv42 = bias_variable([fourth_conv_featuremap])
        h_conv42 = tf.nn.relu(conv2d(h_conv41, w_conv42) + b_conv42)

        w_conv43 = weight_variable([filter_size, filter_size, fourth_conv_featuremap, fourth_conv_featuremap])
        b_conv43 = bias_variable([fourth_conv_featuremap])
        h_conv43 = tf.nn.relu(conv2d(h_conv42, w_conv43) + b_conv43)

        maxpool4 = max_pool_overlap_2x2(h_conv43)

    with tf.name_scope('conv5'):
        w_conv51 = weight_variable([filter_size, filter_size, fourth_conv_featuremap, fifth_conv_featuremap])
        b_conv51 = bias_variable([fifth_conv_featuremap])
        h_conv51 = tf.nn.relu(conv2d(maxpool4, w_conv51) + b_conv51)

        w_conv52 = weight_variable([filter_size, filter_size, fifth_conv_featuremap, fifth_conv_featuremap])
        b_conv52 = bias_variable([fifth_conv_featuremap])
        h_conv52 = tf.nn.relu(conv2d(h_conv51, w_conv52) + b_conv52)

        w_conv53 = weight_variable([filter_size, filter_size, fifth_conv_featuremap, fifth_conv_featuremap])
        b_conv53 = bias_variable([fifth_conv_featuremap])
        h_conv53 = tf.nn.relu(conv2d(h_conv52, w_conv53) + b_conv53)

        maxpool5 = max_pool_overlap_2x2(h_conv53)

    # local3
    with tf.variable_scope('fc1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(maxpool5, [-1, 7*7*512])
        weights = weight_variable(shape=[7*7*512, 4096],name='weights_fc1',)
        biases = bias_variable([4096],'biases_fc1')
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('fc2') as scope:
        weights = weight_variable(shape=[4096, 4096],name='weights_fc2')
        biases = bias_variable([4096],'biases_fc2')
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        # keep_prob_local4 = tf.placeholder(tf.float32)
        # local4_drop = tf.nn.dropout(local4, keep_prob_local4)
    #local4
    with tf.variable_scope('fc3') as scope:
        weights = weight_variable(shape=[4096, 10],name='weights_fc3')
        biases = bias_variable([10],'biases_fc2')
        softmax_linear = tf.add(tf.matmul(fc2, weights),biases)

    return softmax_linear


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
    x = tf.placeholder(tf.float32, shape=[None, src_size, src_size, img_channel], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    y_cls = tf.argmax(y_, dimension=1)

    #build the graph
    y_cnn = inference(x)

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

        except Exception as e:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Restore fails : {0}, initialize...".format(e))

            init = tf.global_variables_initializer()
            sess.run(init)



        #record the process
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

        num_iterations = int(training_size/batch_size)


        for i in xrange(epoch):
            images,labels = shuffle(src_images,src_labels)

            for j in xrange(num_iterations):
                step = start_step + i*num_iterations + j +1


                #begin to train now. First load the data
                x_batch,y_batch = sequence_batch(images,labels,j,batch_size)

                _, loss_value,_summary_str =sess.run([train_step,loss,summary],
                    feed_dict={
                        x: x_batch,
                        y_: y_batch
                    }
                )

                summary_writer.add_summary(_summary_str, step)

                # limited memory, accuracy is calculated with a batch of 5000, and count mean value
                accuracy_batch = 200
                count_times = int(training_size/accuracy_batch)
                train_accuracy = 0


                if(j/100==0):
                    print("current step:{0}".format(step))
                if (j == num_iterations - 1):

                    for k in xrange(count_times):
                        accuracy_x_batch, accuracy_y_batch = sequence_batch(images, labels, k, accuracy_batch)

                        training_feed_dict = {
                            x: accuracy_x_batch,
                            y_: accuracy_y_batch
                        }

                        train_accuracy = train_accuracy + accuracy.eval(feed_dict=training_feed_dict)
                    #get mean value
                    train_accuracy =train_accuracy/count_times
                    print('step %d, training accuracy %g' %(step, train_accuracy))


                    test_count_times = int(test_size / accuracy_batch)
                    test_accuracy = 0

                    for test_k in xrange(test_count_times):
                        test_accuracy_x_batch, test_accuracy_y_batch \
                            = sequence_batch(src_test_images, src_test_labels, test_k, accuracy_batch)

                        test_feed_dict = {
                            x: test_accuracy_x_batch,
                            y_: test_accuracy_y_batch
                        }


                        test_accuracy = test_accuracy + accuracy.eval(feed_dict=test_feed_dict)
                    #calculate the total test accuracy
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



def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run(main=main)