import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

# create two variable for sigmoid and relu respectively
# Outputs random values from a normal distribution.
a1=tf.Variable(tf.random_normal(shape=[1,1]))
# Outputs random values from a uniform distribution.
#shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
b1=tf.Variable(tf.random_uniform(shape=[1,1]))

a2=tf.Variable(tf.random_normal(shape=[1,1]))
b2=tf.Variable(tf.random_uniform(shape=[1,1]))

a3=tf.Variable(tf.random_normal(shape=[1,1]))
b3=tf.Variable(tf.random_uniform(shape=[1,1]))

print('>>>>',a1)
print('>>>>',b1)
print('>>>>',a2)
print('>>>>',b2)

# random samples from a normal (Gaussian) distribution
# loc=Mean (centre) of the distribution
# scale=Standard deviation (spread or width) of the distribution
# size=Output shape
x=np.random.normal(2,0.1, 500)

#create placeholder
x_data=tf.placeholder(dtype = tf.float32,shape=(50, 1))

batch_size = 50

sigmoid_activation=tf.sigmoid(tf.add(tf.matmul(x_data,a1),b1))

tanh_activation=tf.tanh(tf.add(tf.matmul(x_data,a3),b3))

relu_activation=tf.nn.relu(tf.add(tf.matmul(x_data,a2),b2))
# tf.reduce_mean=Computes the mean of elements across dimensions of a tensor
loss1=tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2=tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))
loss3=tf.reduce_mean(tf.square(tf.subtract(tanh_activation, 0.75)))

sess=tf.Session()

init=tf.global_variables_initializer()
sess.run(init)

my_opt=tf.train.GradientDescentOptimizer(0.01)
train_step_sigmoid=my_opt.minimize(loss1)
train_step_relu=my_opt.minimize(loss2)
train_step_tanh=my_opt.minimize(loss3)
loss_vac_sigmoid=[]
loss_vec_relu=[]
loss_vec_tanh=[]

for i in range(500):
    rand_indices=np.random.choice(len(x),size=batch_size)
    x_vals=np.transpose([x[rand_indices]])
    

    sess.run(train_step_sigmoid,feed_dict={x_data: x_vals})
    sess.run(train_step_relu,feed_dict={x_data: x_vals})
    sess.run(train_step_tanh,feed_dict={x_data: x_vals})

    loss_vac_sigmoid.append(sess.run(loss1,feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(loss2,feed_dict={x_data: x_vals}))
    loss_vec_tanh.append(sess.run(loss2,feed_dict={x_data: x_vals}))
    
    sigmoid_output=np.mean(sess.run(sigmoid_activation,feed_dict={x_data: x_vals}))
    relu_output=np.mean(sess.run(relu_activation,feed_dict={x_data: x_vals}))
    tanh_output=np.mean(sess.run(tanh_activation,feed_dict={x_data: x_vals}))
    if i %50 == 0:
        print ('sigmoid = ' + str(np.mean(sigmoid_output)))
        print ('relu = ' + str(np.mean(relu_output)))
        print ('tanh = ' + str(np.mean(tanh_output)))

x = np.linspace(0, 1, 100)
plt.plot(x, loss_vac_sigmoid, label='sigmoid')
plt.plot(x, loss_vec_relu, label='relu')
plt.plot(x, loss_vec_tanh, label='tanh')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()

