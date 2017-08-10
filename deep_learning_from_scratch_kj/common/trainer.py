# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *

write = sys.stdout.write

class Trainer:
	"""ニューラルネットの訓練を行うクラス
	"""
	def __init__(self, network, x_train, t_train, x_test, t_test,
				 epochs=20, flg_batch_online='mini_batch', mini_batch_size=100,
				 optimizer='SGD', optimizer_param={'lr':0.01}, 
				 evaluate_sample_num_per_epoch=None, verbose=True):
		self.network = network
		self.verbose = verbose
		self.x_train = x_train
		self.t_train = t_train
		self.x_test = x_test
		self.t_test = t_test
		self.epochs = epochs
		self.flg_batch_online = flg_batch_online
		self.batch_size = mini_batch_size
		self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

		# optimzer
		optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
								'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
		self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
		
		self.train_size = x_train.shape[0]
		self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
		self.max_iter = int(epochs * self.iter_per_epoch)
		self.current_iter = 0
		self.current_epoch = 0
		
		self.train_loss_list = []
		self.train_acc_list = []
		self.test_acc_list = []
		
	def init_statistics(self, num_layers):
		self.total_loss_list = []
		self.total_loss = 0
		
		self.grad_norm_list = []
		
		self.num_layers = num_layers
		# variables for statistics (activation distribution)
		self.out_labels = []
		for idx in range(self.num_layers-1):
			self.out_labels.append('Activation_function'+str(idx+1))
	
		self.out_value_by_layer = {}
		for label in self.out_labels:
			self.out_value_by_layer[label] = []

	def train_step(self):
		if self.flg_batch_online == 'online':
			idx_train = self.current_iter % self.train_size
			x_batch = self.x_train[idx_train]
			t_batch = self.t_train[idx_train]
		else:
			batch_mask = np.random.choice(self.train_size, self.batch_size)
			x_batch = self.x_train[batch_mask]
			t_batch = self.t_train[batch_mask]
			
		if t_batch.size == 1:
			t_batch = np.array([t_batch])
		
		grads = self.network.gradient(x_batch, t_batch)
		
		# store activation distribution
		if self.current_iter >= self.max_iter - self.train_size:
			for label in self.out_labels:
				self.out_value_by_layer[label].append(self.network.get_out_value(label))
		# accumulate loss value
		self.total_loss += self.network.this_loss

		self.optimizer.update(self.network.params, grads)
		
		loss = self.network.loss(x_batch, t_batch)
		self.train_loss_list.append(loss)
		if self.verbose: print("train loss:" + str(loss))
		
		if self.current_iter % self.iter_per_epoch == 0:
			self.current_epoch += 1
			
			if self.evaluate_sample_num_per_epoch is None:
				x_train_sample, t_train_sample = self.x_train, self.t_train
				x_test_sample, t_test_sample = self.x_test, self.t_test
			else:
				t = self.evaluate_sample_num_per_epoch
				x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
				x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
				
			train_acc = self.network.accuracy(x_train_sample, t_train_sample)
			test_acc = self.network.accuracy(x_test_sample, t_test_sample)
			'''
			sys.stderr.write('###### '+str(self.current_iter)+'-th iteration ######\n')
			print('###### ', self.current_iter, '-th iteration ######')
			print('Confusion Matrix: Train')
			self.network.print_parameter(self.network.confusion_matrix)

			print('Confusion Matrix: Test')
			self.network.print_parameter(self.network.confusion_matrix)
			'''

			self.train_acc_list.append(train_acc)
			self.test_acc_list.append(test_acc)
			print(self.current_iter, train_acc, test_acc, loss, self.total_loss)
			#if self.current_iter > 0:
			self.total_loss_list.append(self.total_loss)
			self.total_loss = 0
			self.grad_norm_list.append(self.network.grad_norm)

			if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
		#self.current_iter += 1

	def train(self):
		for self.current_iter in range(self.max_iter):
			self.train_step()

		test_acc = self.network.accuracy(self.x_test, self.t_test)

		if self.verbose:
			print("=============== Final Test Accuracy ===============")
			print("test acc:" + str(test_acc))

