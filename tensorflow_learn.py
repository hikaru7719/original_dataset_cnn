import tensorflow as tf

NUM_CLASS = 2


def read_csv(csvfile):
    fname_queue = tf.train.string_input_producer([csvfile])
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])
    label = tf.cast(label, tf.int64)
    label = tf.one_hot(label, depth = 2, on_value = 1.0, off_value = 0.0, axis = -1)
    jpeg = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.cast(image, tf.float32)
    image.set_shape([64,64, 3])
    return  _generate_image_and_label_batch(image,label,fname, min_queue_examples=1000, batch_size=32,shuffle=True)


def _generate_image_and_label_batch(image, label, filename, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    capacity = min_queue_examples + 3 * batch_size
    if shuffle:
        images, label_batch, filename = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, filename = tf.train.batch(
            [image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('image', images, 100)

    labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])
    return images, labels, filename


def forword_output(x):
    x_image = tf.reshape(x, [-1, 64, 64, 3])
    print(x_image)

    with tf.name_scope('conv1') as scope:
        f1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32]))
        conv1 = tf.nn.conv2d(x_image, f1, strides=[1,1,1,1], padding='SAME')
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

    with tf.name_scope('conv3') as scope:
        f3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128]))
        conv3 = tf.nn.conv2d(h_pool2, f3, strides=[1,1,1,1], padding='SAME')
        b3 = tf.Variable(tf.constant(0.1, shape=[128]))
        h_conv3 = tf.nn.relu(conv3 + b3)
        tf.summary.histogram('f3',f3)

    with tf.name_scope('pool3') as scope:
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        print(h_pool3)

    with tf.name_scope('flat') as scope:
        h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*128])

    with tf.name_scope('fc1') as scope:
        w_fc1 = tf.Variable(tf.truncated_normal([8*8*128,1024],stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1,shape = [1024]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1)

    with tf.name_scope('fc2') as scope:
        w_fc2 = tf.Variable(tf.truncated_normal([1024, 2]))
        b_fc2 = tf.Variable(tf.constant(0.1,shape = [2]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        print(h_fc2)

    with tf.name_scope('softmax') as scope:
        out = tf.nn.softmax(h_fc2)
        print(out)

    return out

def loss_func(labels,out):
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(out + 1e-5), axis=[1]))



def main():
    csv_name = 'train.csv'
    images,labels,filename = read_csv(csv_name)

    out = forword_output(images)

    loss = loss_func(labels,out)

    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

    correct = tf.equal(tf.argmax(out,1), tf.argmax(labels,1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    summry_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        summry_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(init)
        tf.train.start_queue_runners(sess)

        for i in range(10000):
            step = i + 1
            # x = sess.run(images)
            # print(x)
            sess.run(train_step)
            acc_val = sess.run(accuracy)

            if step % 10 == 0:
                summry_str = sess.run(summry_op)
                summry_writer.add_summary(summry_str,step)
                print(acc_val)



if __name__ == '__main__':
    main()
