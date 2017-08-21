#a least procedure to test the max_to_keep work or not
import tensorflow as tf
import os
cos_zero = tf.constant(0)
v_zero = tf.Variable(cos_zero)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver(max_to_keep=4)
log_dir = os.path.join(os.getcwd(),'saver_test')
ch_file = os.path.join(log_dir,'model.ckpt')
print(ch_file)

for i in range(4):
    saver.save(sess,save_path=ch_file,global_step=i)

last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=log_dir)
print(last_chk_path)

last_chk = last_chk_path.split('/')[-1]
filelist = [ f for f in os.listdir(log_dir) \
             if f.startswith('model.ckpt') and not f.startswith(last_chk)]

print(filelist)

for f in filelist:
    os.remove(os.path.join(log_dir,f))


