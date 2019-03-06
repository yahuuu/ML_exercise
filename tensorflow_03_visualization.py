import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):

    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),
                                  name='Weithtssss')

        with tf.name_scope('biase'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,
                                 name='bias')

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#建立数据

#下面这段代码等价于 x_data = np.linspace(-1, 1, 300)[:, None]
#增加一个维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# 输入数据保留,定义框架
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                        reduction_indices=[1])
                          )

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess= tf.Session()
#写到文件中，浏览器打开
writer = tf.summary.FileWriter("tensorflow_logs/",sess.graph)
writer.close()



# important step
sess.run(init)

# # plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.ioff()
#
# plt.show()

#
# for i in range(1000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         # to visualize the result and improvement
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#         # plot the prediction
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#         plt.pause(1)
