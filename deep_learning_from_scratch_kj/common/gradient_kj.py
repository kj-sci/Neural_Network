# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)

	print('----- 1d ------')
	print(x)
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)
		print("x[",idx,"]=", x[idx])
		x[idx] = float(tmp_val) - h 
		print("x[",idx,"]=", x[idx])
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		print(idx, "|", x, "|", h, "|", fxh1, "|", fxh2, "|", grad[idx])
		x[idx] = tmp_val # 値を元に戻す
		
	return grad


def numerical_gradient_2d(f, X):
	if X.ndim == 1:
		return _numerical_gradient_1d(f, X)
	else:
		grad = np.zeros_like(X)
		
		for idx, x in enumerate(X):
			grad[idx] = _numerical_gradient_1d(f, x)

			print("-------- 2d ------------")
			print(idx, "|", x, "|", grad[idx])
			
		return grad


def numerical_gradient(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)
	
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		print('----- numerical_grad -----')
		print(idx, "|", x)
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)
		print(idx, "|", x)
		
		x[idx] = tmp_val - h 
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		
		x[idx] = tmp_val # 値を元に戻す
		it.iternext()   
		
	return grad
	
def this_f(this_x):
	return this_x[0]**2+2*this_x[1]**2
	
	
def main():
	this_x = np.array([[0,1], [1,1], [2,2]], dtype=np.float)
	print('-----------------')
	print(this_x)
	grad = numerical_gradient(this_f, this_x)
	print('-----------------')
	print(grad)

if __name__ == '__main__':
	main()