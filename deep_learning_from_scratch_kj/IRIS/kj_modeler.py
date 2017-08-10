# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from data_handler import data_handler
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

import matplotlib.pyplot as plt

write = sys.stdout.write

class kj_modeler:
	def __init__(self):
		# hyper parameter of the neural network
		# num_layers: same definition as PRML (N layers = input + (N-1) hidden + output)
		self.num_layers = 3
		self.input_size = 4
		self.hidden_size_list = [6, 6]
		self.output_size = 3
		# 'sigmoid' / 'relu'
		self.activation = 'sigmoid' #'relu'
		# 0.01 / 'xaivier' or 'sigmoid' / 'he' or 'relu'
		self.weight_init_std = 0.01 # 'xaivier' 'relu'
		
		# parameter labels (W, b)
		self.parameter_labels = []
		for idx in range(self.num_layers):
			self.parameter_labels.append('W'+str(idx+1))
			self.parameter_labels.append('b'+str(idx+1))

		# parameters for simulation
		#self.iters_num = 1001 (3) / 7001 (4) / 10001 (5) / 25001 (6)
		self.num_epochs = 7001
		self.batch_size = 1
		#'sgd' / 'momentum' / 'adagrad'  / 'adam'
		self.optimizer = 'sgd'
		self.learning_rate = 0.01
			
	# read training / test data
	def read_data(self, training_data_fname, test_data_fname, meta_data_fname):
		# Training Data
		self.train_hd = data_handler()
		self.train_hd.process(meta_data_fname, training_data_fname)
		self.x_train = self.train_hd.get_explanatory()
		self.t_train = self.train_hd.get_target()

		self.train_size = self.x_train.shape[0]
		self.iter_per_epoch = max(self.train_size / self.batch_size, 1)

		# Test Data
		self.test_hd = data_handler()
		self.test_hd.process(meta_data_fname, test_data_fname)
		self.x_test = self.test_hd.get_explanatory()
		self.t_test = self.test_hd.get_target()

	# define neural network
	def setup_neural_network(self):
		self.network = MultiLayerNetExtend(input_size=self.input_size, hidden_size_list=self.hidden_size_list,
									  output_size=self.output_size, activation=self.activation, weight_init_std=self.weight_init_std)
		self.trainer = Trainer(self.network, self.x_train, self.t_train, self.x_test, self.t_test,
				  epochs=self.num_epochs, flg_batch_online = 'online', mini_batch_size=self.batch_size,
				  optimizer=self.optimizer, optimizer_param={'lr': self.learning_rate}, verbose=False)

		self.trainer.init_statistics(self.num_layers)
		
	def training(self):
		self.trainer.train()

	def print_parameter_value(self, header):
		print(header)
		self.network.print_all_parameters()

	def plot_loss_hist(self):
		self.total_loss_list = np.array(self.trainer.total_loss_list)
		print("Loss Hist:")
		for loop, row in enumerate(self.total_loss_list):
			print(loop,'\t',row)
		plt.plot(np.arange(self.total_loss_list.size), self.total_loss_list)
		plt.show()
		
	def plot_grad_hist(self):
		print("Grad Norm Hist:")

		n_iter = len(self.trainer.grad_norm_list)
		grad_norm = np.zeros(n_iter)
		for idx_layer in range(self.num_layers):
			label = 'W'+str(idx_layer+1)
			for iter in range(n_iter):
				grad_norm[iter] = self.trainer.grad_norm_list[iter][label]

			plt.subplot(self.num_layers, 1, idx_layer+1)
			plt.plot(np.arange(n_iter), grad_norm)
			plt.title(label)
		
		plt.show()


	def plot_activation_dist(self):
		# Activation Distribution
		idx = 1
		for label in self.trainer.out_labels:
			plt.subplot(len(self.trainer.out_labels), 1, idx)
			out_value = np.array(self.trainer.out_value_by_layer[label])
			plt.hist(out_value.flatten(), 30, range=(0,1))
			plt.title(label)
			idx += 1

		#plt.subplot(1, 2, 2)
		#out_value = np.array(self.out_value_by_layer['Sigmoid2'])
		#plt.hist(out_value.flatten(), 30, range=(0,1))
		#plt.title('Sigmoid2')

		'''
		print('Out Value (Sigmoid1)')
		for idx1 in range(self.out_value_by_layer.shape[0]):
			write(str(idx1)+":\t")
			for idx2 in range(self.out_value_by_layer.shape[1]):
				if idx2 > 0:
					write('\t')
				write(str(self.out_value_by_layer[idx1][idx2]))
			write('\n')
		'''

		plt.show()

def main():
	training_data_fname = 'iris_data_train.txt'
	test_data_fname = 'iris_data_test.txt'
	meta_data_fname = 'iris_metadata.txt'

	hd = kj_modeler()
	hd.read_data(training_data_fname, test_data_fname, meta_data_fname)
	hd.setup_neural_network()
	hd.print_parameter_value('===== Init Param Value =======')
	hd.training()
	#hd.plot_loss_hist()
	hd.plot_grad_hist()
	#hd.plot_activation_dist()
	
	
if __name__ == '__main__':
	main()
