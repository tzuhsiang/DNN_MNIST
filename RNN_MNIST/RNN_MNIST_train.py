import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# Parameters

tf.app.flags.DEFINE_float("lr", 0.0017,"learning_rate")
tf.app.flags.DEFINE_integer("batch", 128,  "Batch size to use during training.")
tf.app.flags.DEFINE_integer("step", 5000, "training_iters")
tf.app.flags.DEFINE_boolean("GPU",True,"Using GPU or not")


FLAGS = tf.app.flags.FLAGS


learning_rate = FLAGS.lr
batch_size = FLAGS.batch
training_iters = FLAGS.step
display_step = 1000
Use_GPU=FLAGS.GPU




# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


def train():

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    def RNN(x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)

        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))
    sess.run(init)
    step = 1

    t1=time.time()


    while step  < (training_iters+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    t2=time.time()
    print("Optimization Finished!")
    print("Execution time:%f"%(t2-t1))


    #test_len = 10000
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    print("Execution time: %.2f sec"%(time.time()-t1))

    with tf.device('/cpu:0'):
        saver =  tf.train.Saver(tf.global_variables())
        model_path="./Models/rnn_mnist"
        saver.save(sess=sess, save_path=model_path)
        print("Saved the model %s" %model_path)



def main(_):
    time_start=time.time()
    
    if Use_GPU:
        device='/gpu:0'
    else:
        device='/cpu:0'
        
        
    with tf.device(device):
        train()
    
    time_end=time.time()
    print("Execution:%d Sec"%(time_end-time_start))

if __name__ == "__main__":
    tf.app.run()





    