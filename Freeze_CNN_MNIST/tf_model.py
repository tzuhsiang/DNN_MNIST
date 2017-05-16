import tensorflow as tf

import os
#dir = os.path.dirname(os.path.realpath(__file__))

# First, you design your mathematical operations
# We are the default graph scope


tf.app.flags.DEFINE_string("train_dir",'./models/',"model direction")
FLAGS = tf.app.flags.FLAGS


train_dir = FLAGS.train_dir


# Let's design a variable
with tf.name_scope('Model'):
    v1= tf.Variable(1. , name="v1")
    v2 = tf.Variable(2. , name="v2")
    # Let's design an operation
    a = tf.add(v1, v2, name="add")


saver = tf.train.Saver() 

# By default the Session handles the default graph and all its included variables
with tf.Session() as sess:  
  sess.run(tf.global_variables_initializer())

  saver.save(sess, train_dir+"simple_model")
