import tensorflow as tf
import time


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


# Parameters

tf.app.flags.DEFINE_string("model_path",'./models/cnn_mnist',"model direction")
tf.app.flags.DEFINE_float("lr", 0.0017,"learning_rate")
tf.app.flags.DEFINE_integer("batch", 128,  "Batch size to use during training.")
tf.app.flags.DEFINE_integer("step", 5000, "training_iters")
tf.app.flags.DEFINE_boolean("GPU",True,"Using GPU or not")


FLAGS = tf.app.flags.FLAGS

model_path=FLAGS.model_path
learning_rate = FLAGS.lr
batch_size = FLAGS.batch
training_iters = FLAGS.step
display_step = 1000
Use_GPU=FLAGS.GPU




# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


def train(Model_Path=model_path):

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input],name="x")
    y = tf.placeholder(tf.float32, [None, n_classes],name="y")
    keep_prob = tf.placeholder(tf.float32,name="keep") #dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    with tf.name_scope('Model'):
        # Construct model
        pred = conv_net(x, weights, biases, keep_prob)
        
    with tf.name_scope('Loss'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    
    with tf.name_scope('Accuracy'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="acc")
    
    with tf.device('/cpu:0'):
        
        # Create a summary to monitor cost tensor
        tf.summary.scalar("Loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("Accuracy", accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()    
    
    summary_writer = tf.summary.FileWriter(Model_Path, graph=tf.get_default_graph())
    
    # Initializing the variables
    init = tf.global_variables_initializer()


    t1=time.time()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # Launch the graph
    sess.run(init)
    
    
    
    step = 1
    # Keep training until reach max iterations
    while step  < (training_iters)+1:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        _, summary=sess.run([optimizer,merged_summary_op],feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        summary_writer.add_summary(summary, step)
        
        if (step % display_step == 0):
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y,keep_prob: 1.})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    
    # Calculate accuracy for 10000 mnist test images
    #print("Testing Accuracy:", \
    #       sess.run(accuracy, feed_dict={x: mnist.test.images[:],y: mnist.test.labels[:], keep_prob: 1.}))

    print("Execution time: %.2f sec"%(time.time()-t1))

    with tf.device('/cpu:0'):
        saver =  tf.train.Saver(tf.global_variables())
        saver.save(sess=sess, save_path=Model_Path)
        print("Saved the model %s" %Model_Path)
        print("Run the command line:\n--> tensorboard --logdir=%s"%Model_Path)


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





    