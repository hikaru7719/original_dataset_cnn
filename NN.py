from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
img = tf.reshape(x,[-1,28,28,1])
tf.summary.image("input_data", img, 10)

with tf.name_scope("hidden"):
    w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name="w1")
    b_1 = tf.Variable(tf.zeros([64]), name="b1")
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

    tf.summary.histogram('w_1',w_1)
with tf.name_scope("output"):
    w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
    b_2 = tf.Variable(tf.zeros([10]), name="b2")
    out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

    tf.summary.histogram('w_2',w_2)

y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - out))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)

summry_op = tf.summary.merge_all()

with tf.Session() as sess:

    summry_writer = tf.summary.FileWriter("./logs", sess.graph)
    sess.run(init)

    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for i in range(5000):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

        if step % 10 == 0:
            summry_str = sess.run(summry_op, feed_dict={x:test_images, y:test_labels})
            summry_writer.add_summary(summry_str, step)
