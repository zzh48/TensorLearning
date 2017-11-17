import tensorflow as tf
from numpy.random import RandomState

#define trainning data batch size
batch_size = 8

#define weights
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

#define placeholder
x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y-input')


#define front_propagation
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#define loss function and back_propagation 
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


#build a simulate data set
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

#x1+x2 < 1 assume as a positive sample
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

#create session for tensorflow
with tf.Session() as sess:
	#initialize variables
	sess.run(tf.global_variables_initializer())

	#print out weights before trainning
	print sess.run(w1)
	print sess.run(w2)

	#set up trainning rounds
	STEPS = 5000
	for i in range(STEPS) :
		#select batch_size of data set to train at each step
		start = (i *  batch_size) % dataset_size
		end = min(start + batch_size, dataset_size) 

		#update parameter input
		sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})
		if i % 1000 == 0 :
			#calculate cross entropy and output
			total_cross_entropy = sess.run(cross_entropy, feed_dict = {x:X, y_:Y})
			print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

	print sess.run(w1)
	print sess.run(w2)