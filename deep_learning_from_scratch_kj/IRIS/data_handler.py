# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
#import re
#import json

write = sys.stdout.write

class data_handler:
	def __init__(self):
		self.data = None
		
		self.metadata = {}

	# wrapper
	def process(self, metadata_fname, data_fname):
		self.read_metadata(metadata_fname, 'cp932')
		self.read_data(data_fname, 'Y', '\t')
		self.split_cols()
		
	def read_metadata(self, metadata_fname, metadata_encoding):
		fp = open(metadata_fname, 'r', encoding=metadata_encoding)
		
		flg = 0
		while(flg == 0):
			line = fp.readline()
			
			if line[0] == '#':
				# meta data
				data = line[:-1].split('\t')
				self.metadata[data[1]] = data[2]
			else:
				# header
				flg = 1
		
		# dtype
		for line in fp:
			if 'dtype' not in self.metadata:
				ptr = {}
				self.metadata['dtype'] = ptr
						
			data = line[:-1].split('\t')
			if data[1] == 'int':
				ptr[data[0]] = int
			elif data[1] == 'float':
				ptr[data[0]] = float
			else:
				sys.stderr.write('Error, dtype unknown\n')
				sys.stderr.write(data[0]+"\t"+data[1]+'\n')
		fp.close()

	def read_data(self, data_fname, flg_header, delimiter):
		self.metadata['delimiter'] = delimiter
		if flg_header == 'N':
			self.metadata['header'] = 'N'
		self.data = pd.read_csv(data_fname, **self.metadata)
		
	def split_cols(self):
		target_idx = 0
		self.target = np.array(self.data.iloc[:,[target_idx]]).flatten()
		start_idx = target_idx+1
		self.explanatory = np.array(self.data.iloc[:,start_idx:])
		#print("---- target ----")
		#print(self.target)
		#print("---- explanatory ----")
		#print(self.explanatory)
		
		#w = np.array([[1], [1], [1], [1]])
		#ans = np.dot(self.explanatory, w)
		#print(ans)

	def get_target(self):
		return self.target
	
	def get_explanatory(self):
		return self.explanatory
		
	# dame
	#def metadata2json(self, json_fname):
	#	fp = open(json_fname, 'w')
	#	json.dump(self.metadata, fp)
	#	fp.close()
		
	def output(self, out_data_fname, delimiter):
		self.data.to_csv(out_data_fname, sep=delimiter, index=False) 
	
def main():
	if len(sys.argv) < 5:
		sys.stderr.write('Usage: python $0 data_filename Y(with header)/N(without header) metadata_filename out_filename\n')
		sys.exit(1)
	
	data_fname = sys.argv[1]
	flg_header = sys.argv[2]
	metadata_fname = sys.argv[3]
	out_fname = sys.argv[4]
	
	delimiter = '\t'
	hd = data_handler()

	metadata_encoding = 'cp932'
	hd.read_metadata(metadata_fname, metadata_encoding)
	#hd.metadata2json('iris_metadata.json.txt')

	hd.read_data(data_fname, flg_header, delimiter)
	hd.split_cols()
	
	hd.output(out_fname, delimiter)
	
if __name__ == '__main__':
	main()


		