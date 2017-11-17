import tensorflow as tf

#declare variable w1, w2. parameter seed defs random seed
#same random seed makes result same
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1), name = "w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1), name = "w2")

#temp define x as a constant
x = tf.constant([0.7, 0.9])
#x = tf.Variable(tf.random_normal([2], stddev = 1, seed = 1), name = "x")
#x = tf.placeholder(tf.float32, shape = (1, 2), name = "input")

tf.assign(w1, w2, validate_shape = False)
#calculate the output 
a = tf.matmul(x, w1) #Get Error: ValueError: Shape must be rank 2 but is rank 1 for 'MatMul' (op: 'MatMul') with input shapes: [2], [2,3].
y = tf.matmul(a, w2)


sess = tf.Session()
#sess.run(w1.initializer)
#sess.run(w2.initializer)
sess.run(tf.global_variables_initializer())

print(sess.run(y))
#print(sess.run(y, feed_dict = {x: [[0.7, 0.9]]}))
sess.close()