import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# example: Number

op1 = tf.constant(1, name="op1")
op2 = tf.constant(2, name="op2")

plus = tf.add(op1, op2)

with tf.Session() as session:
    print(session.run(plus))


op3 = tf.Variable(3, name="op3")
value = tf.add(op1, op3)
update = tf.assign(op3, value)
#op3 = value

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(update))
    print(session.run(update))


placeholder = tf.placeholder(tf.int32, name="holder")

plusholder = tf.add(op1, placeholder)

with tf.Session() as session:
    print(session.run(plusholder, feed_dict={placeholder:15}))

with tf.Session() as session:
    print(session.run(plusholder, feed_dict={placeholder:100}))

print()


# example: Tensor

tensor1 = tf.constant([[1, 2], [3, 4]], tf.int32, name="tensor1")
tensor2 = tf.Variable([[3, 5], [6, 8]], tf.int32, name="tensor2")

#plustensor = tf.add(tensor1, tensor2)
plustensor = tensor1 + tensor2
#multitensor = tf.multiply(tensor1, tensor2)
multitensor = tensor1 * tensor2
updatetensor = tf.assign(tensor2, multitensor)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print("plus : ------------------")
    print(session.run(plustensor))
    print()

    print("multi : -----------------")
    print(session.run(multitensor))
    print()

    print("update : ----------------")
    for i in range(3):
        print(session.run(updatetensor))

