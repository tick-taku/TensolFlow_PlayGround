import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prepare

x_data = np.linspace(0., 1., 6)
a_answer = 1.5
b_answer = .1
y_data = a_answer * x_data + b_answer


# Define

x_answer = tf.placeholder(tf.float32)
y_answer = tf.placeholder(tf.float32)

a_model = tf.Variable(1.0)
b_model = tf.Variable(0.0)

y_model = a_model * x_answer + b_model

loss = tf.sqrt(tf.reduce_mean((y_model - y_answer)**2))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)


# Run

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(20000):
        session.run(train, {x_answer: x_data, y_answer: y_data})

        if i % 1000 == 0:
            current_loss, current_y_model = session.run([loss, y_model], {x_answer: x_data, y_answer: y_data})
            print("Loss: {current_loss}")
            print("y_model: {current_y_model}, y_answer: {y_data}")
