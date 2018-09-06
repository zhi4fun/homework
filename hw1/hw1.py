"""
Code for homework 1
Author: Zhi Chen
Example Usage: change the section for different questions (2-2, 2-3, 3-2)
	python hw1.py --sec 22 --env "Ant-v2"
	python hw1.py --sec 23
	python hw1.py --sec 32
"""


import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import argparse
import matplotlib.pyplot as plt

# create a fully connected layer
def fully(x, xdim, ydim, name='Fully'):
	with tf.variable_scope(name):
		matrix = tf.get_variable('Matrix', [xdim, ydim], tf.float32, tf.random_normal_initializer(stddev=0.1))
		bias = tf.get_variable('Bias', [ydim], tf.float32, tf.constant_initializer(0.0))
		return tf.matmul(x, matrix) + bias

# create a neural net with 3 hidden layers
def nn(x, xdim, udim, hdim, name):
	with tf.variable_scope(name):
		h0 = tf.nn.tanh(fully(x, xdim, hdim[0], name='Fully0'))
		h1 = tf.nn.tanh(fully(h0, hdim[0], hdim[1], name='Fully1'))
		h2 = tf.nn.tanh(fully(h1, hdim[1], hdim[2], name='Fully2'))
		u = fully(h2, hdim[2], udim, name='Fully3')
		return u

def run_expert(envname, rollouts, render, maxsteps=0):
	policy_fn = load_policy.load_policy('experts/' + envname + '.pkl')
	with tf.Session():
		tf_util.initialize()

		env = gym.make(envname)
		max_steps = maxsteps or env.spec.timestep_limit
		returns = []
		observations = []
		actions = []

		for i in range(rollouts):
			print('Expert rollout ' + str(i))
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = policy_fn(obs[None,:])
				observations.append(obs)
				actions.append(action[0,:])
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1
				if render:
					env.render()
				if steps >= max_steps:
					break
			returns.append(totalr)
		return np.array(observations), np.array(actions), np.mean(returns), np.std(returns)

def model(envname, name, hdim):
	env = gym.make(envname)
	xdim = env.observation_space.shape[0]
	udim = env.action_space.shape[0]
	x = tf.placeholder(tf.float32, [None, xdim], name='Observation')
	u_ = tf.placeholder(tf.float32, [None, udim], name='Action')
	u = nn(x, xdim, udim, hdim, name)
	loss = tf.nn.l2_loss(u - u_)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
	print('Model built.')
	return x, u_, u, loss, train_step

def train(sess, x, u_, u, loss, train_step, batch_size, epochs, expert_obs, expert_act):
	print('Start training ...')
	loss_seq = []
	samples = np.amax(expert_obs.shape)
	with sess.as_default():
		for i in range(epochs):
			for j in range(int(samples/batch_size)):
				obs = expert_obs[j*batch_size:(j+1)*batch_size, :]
				act_ = expert_act[j*batch_size:(j+1)*batch_size, :]
				sess.run(train_step, feed_dict={x: obs, u_:act_})
			loss_current = loss.eval({x:obs, u_:act_})
			loss_seq.append(loss_current)
			print('epoch: ' + str(i) + ' | loss: ' + str(loss_current))
		print('Training completed.')
		return sess, loss_seq

def run_trained(envname, rollouts, render, sess, x, u, maxsteps=0):
	with sess.as_default():
		env = gym.make(envname)
		max_steps = maxsteps or env.spec.timestep_limit
		returns = []
		observations = []
		actions = []

		for i in range(rollouts):
			print('Expert rollout ' + str(i))
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				action = sess.run(u, feed_dict={x: obs[None,:]})
				observations.append(obs)
				actions.append(action)
				obs, r, done, _ = env.step(action)
				totalr += r
				steps += 1
				if render:
					env.render()
				if steps >= max_steps:
					break
			returns.append(totalr)
		return np.array(observations), np.array(actions), np.mean(returns), np.std(returns)

def s2_2(env):
	envname = env
	expert_obs, expert_act, expert_mean, expert_std = run_expert(envname, 20, False) 
	hdim = [1024, 512, 128]
	batch_size = 10
	x, u_, u, loss, train_step = model(envname, envname, hdim)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	epochs = 50
	sess, loss_seq = train(sess, x, u_, u, loss, train_step, batch_size, epochs, expert_obs, expert_act)
	_, _, trained_mean, trained_std = run_trained(envname, 20, True, sess, x, u)
	print('mean_r= ' + str(trained_mean) + '| std_r= ' + str(trained_std))

def s2_3(env):
	envname = env
	expert_obs, expert_act, expert_mean, expert_std = run_expert(envname, 20, False) 
	hdim = [1024, 512, 128]
	batch_size = 10
	x, u_, u, loss, train_step = model(envname, envname, hdim)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	epochs = 20
	sess, loss_seq = train(sess, x, u_, u, loss, train_step, batch_size, epochs, expert_obs, expert_act)
	_, _, _, _ = run_trained(envname, 1, False, sess, x, u)
	
	x = np.arange(epochs)
	plt.plot(x, loss_seq)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.show()

def s3_2(env):
	envname = env
	rollouts = 20
	da_iters = 20
	hdim = [1024, 512, 128]
	batch_size = 10
	max_steps = 50
	# expert policy mean and std
	expert_obs, expert_act, expert_mean, expert_std = run_expert(envname, rollouts, False, max_steps) 

	# bc mean and std
	x, u_, u, loss, train_step = model(envname, envname, hdim)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	BCMean = []
	BCStd = []
	for k in range(da_iters):
		sess, loss_seq = train(sess, x, u_, u, loss, train_step, batch_size, 1, expert_obs, expert_act)
		_, _, bc_mean, bc_std = run_trained(envname, rollouts, False, sess, x, u, max_steps)
		BCMean.append(bc_mean)
		BCStd.append(bc_std)

	# dagger mean and std
	x, u_, u, loss, train_step = model(envname, envname+'_d', hdim)
	policy_fn = load_policy.load_policy('experts/' + envname + '.pkl')
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	DAMean = []
	DAStd = []
	for k in range(da_iters):
		sess, loss_seq = train(sess, x, u_, u, loss, train_step, batch_size, 1, expert_obs, expert_act)
		da_obs, da_act, da_mean, da_std = run_trained(envname, rollouts, False, sess, x, u, max_steps)
		with tf.Session():
			da_act = policy_fn(da_obs)
		expert_obs = np.append(expert_obs, da_obs, axis=0)
		expert_act = np.append(expert_act, da_act, axis=0)
		DAMean.append(da_mean)
		DAStd.append(da_std)

	# plot
	x = np.arange(da_iters)
	plt.errorbar(x, DAMean, fmt='o', yerr=DAStd, label='DAgger')
	plt.errorbar(x, BCMean, fmt='o', yerr=BCStd, label='Behavior Cloning')
	plt.errorbar(x, [expert_mean]*da_iters, fmt='o', yerr=expert_std, label='Expert')
	plt.legend()
	plt.xlabel('DAgger_iterations')
	plt.ylabel('Returns')
	plt.show()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sec', type=int)
	parser.add_argument('--env', type=str, default='Ant-v2')
	args = parser.parse_args()
	if args.sec == 22:
		s2_2(args.env)
	elif args.sec == 23:
		s2_3(args.env)
	elif args.sec == 32:
		s3_2(args.env)
	else:
		print('Invalid input.')

if __name__ == '__main__':
	main()
