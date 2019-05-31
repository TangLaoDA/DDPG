import tensorflow as tf


a=tf.Variable([[1],[0],[1]])

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # a=tf.expand_dims(tf.argmax(a,axis=1),axis=1)
    # b=tf.constant(a,dtype=tf.float32)
    print(sess.run(tf.squeeze(tf.one_hot(a,2),axis=1)))