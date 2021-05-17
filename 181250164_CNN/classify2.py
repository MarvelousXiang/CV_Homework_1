import numpy as np
import tensorflow as tf
from PIL import Image

data_path = './train/'
val_path = './val'
model_path = './model'


# 读取数据
def read_data(path):
    datas = []
    labels = []
    for label in range(80):
        for num in range(250):
            img = Image.open('{}{}/{}.jpg'.format(path, label, num))
            data = np.array(img) / 255.0
            datas.append(np.resize(data, (64, 64, 3)))
            labels.append(label)
    return np.array(datas), np.array(labels)


def read_val(path):
    datas = []
    labels = []
    for num in range(10000):
        img = Image.open('{}{}.jpg'.format(path, num))
        data = np.array(img) / 255.0
        datas.append(np.resize(data, (64, 64, 3)))
    with open('./val_anno.txt') as file:
        lines = file.readlines()
        for line in lines:
            labels.append(int(line.split()[1]))
    return np.array(datas), np.array(labels)


# model
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2by2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def global_avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1],
                          strides=[1, 8, 8, 1], padding='SAME')


def norm(x):
    return tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def fc_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    res = tf.matmul(input_layer, W) + b
    return res


x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 80])
hold_prob1 = tf.placeholder(tf.float32)
hold_prob2 = tf.placeholder(tf.float32)

conv_1 = conv_layer(x, shape=[3, 3, 3, 64])
conv_2 = conv_layer(conv_1, shape=[3, 3, 64, 64])
bn_1 = norm(conv_2)
pooling_1 = max_pool_2by2(bn_1)
dropout_1 = tf.nn.dropout(pooling_1, keep_prob=hold_prob1)

conv_3 = conv_layer(dropout_1, shape=[3, 3, 64, 128])
conv_4 = conv_layer(conv_3, shape=[3, 3, 128, 128])
bn_2 = norm(conv_4)
pooling_2 = avg_pool_2by2(bn_2)
dropout_2 = tf.nn.dropout(pooling_2, keep_prob=hold_prob1)

conv_5 = conv_layer(dropout_2, shape=[3, 3, 128, 256])
conv_6 = conv_layer(conv_5, shape=[3, 3, 256, 256])
bn_3 = norm(conv_6)
pooling_3 = global_avg_pool(bn_3)
flat_1 = tf.reshape(pooling_3, [-1, 256])
dropout_3 = tf.nn.dropout(flat_1, keep_prob=hold_prob2)

y_pred = fc_layer(dropout_3, 80)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
Epoch = 10
ind = 0
x_train, temp = read_data(data_path)
out = np.zeros((20000, 80))
out[range(20000), temp] = 1
y_train = out
x_val, temp = read_val(val_path)
out = np.zeros((10000, 80))
out[range(10000), temp] = 1
y_val = out

# 洗牌
arr = np.arange(20000)
np.random.shuffle(arr)
x_train = x_train[arr]
y_train = y_train[arr]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100 * Epoch):
        ind = (ind + 200) % 20000
        batch_x = x_train[ind: ind + 200]
        batch_y = y_train[ind: ind + 200]
        _, loss = sess.run([train, cross_entropy], feed_dict={
            x: batch_x, y_true: batch_y,
            hold_prob1: 0.9, hold_prob2: 0.5})
        print("Epoch:", i // 100 + 1, " ", (i % 100 + 1) * 200, "/ 20000 ", "Loss =", loss)

        if i % 100 == 99:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            acc_mean = sess.run(acc, feed_dict={
                x: x_val, y_true: y_val,
                hold_prob1: 1.0, hold_prob2: 1.0})
            print("\n")
            print("test acc =", acc_mean)
            print('\n')
    saver.save(sess, model_path + 'my_model')

saver = tf.train.Saver()
print('训练完成')
