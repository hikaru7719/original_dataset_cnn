import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def forword_output(x):
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    # print(x_image)

    with tf.name_scope('conv1') as scope:
        f1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
        conv1 = tf.nn.conv2d(x, f1, strides=[1,1,1,1], padding='SAME')
        b1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(conv1 + b1)
        tf.summary.histogram('f1',f1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        print(h_pool1)

    with tf.name_scope('conv2') as scope:
        f2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
        conv2 = tf.nn.conv2d(h_pool1, f2, strides=[1,1,1,1], padding='SAME')
        b2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv2 = tf.nn.relu(conv2 + b2)
        tf.summary.histogram('f2',f2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        print(h_pool2)

    with tf.name_scope('flat') as scope:
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    with tf.name_scope('fc1') as scope:
        w_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1,shape = [1024]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1)

    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal([1024, 10]))
        b_fc2 = tf.Variable(tf.constant(0.1,shape = [10]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        print(h_fc2)

    with tf.name_scope('softmax') as scope:
        out = tf.nn.softmax(h_fc2)
        print(out)

    return out

def loss_func(labels,out):
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(out + 1e-5), axis=[1]))



def main():

    mnist = input_data.read_data_sets("data/", one_hot=True)

    x = tf.placeholder(tf.float32, [50, 784])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, [50, 10])

    out = forword_output(x_image)

    loss = loss_func(y,out)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    summry_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        summry_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(init)
        # tf.train.start_queue_runners(sess)
        test_images = mnist.test.images
        test_labels = mnist.test.labels

        for i in range(10000):
            step = i + 1
            # x = sess.run(images)
            # print(x)
            train_images, train_labels = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x:train_images ,y:train_labels})
            acc_val = sess.run(accuracy)

            if step % 10 == 0:
                summry_str = sess.run(summry_op,feed_dict={x:test_images, y:test_labels})
                summry_writer.add_summary(summry_str,step)



if __name__ == '__main__':
    main()
