''' Test and run tfgen '''

import os
import string
import random
import unittest

import numpy as np

from rast.gen import generator
from rast.parse import parse

from tensor_rast.shaped_node import MAX_RANK, MIN_RANK
from tensor_rast.tfgen import tfgen, traverse

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10

def _randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

def _treeSize(root):
	def count(_, deps):
		return sum(deps) + 1, None
	counter, _ = traverse(root, count)
	return counter

def _randArr(n):
	return [_randVariable(16) for _ in range(n)]

class TestTfgen(unittest.TestCase):
	def test_tfgen(self):
		cfg = open('tensor.yaml', 'r')
		terms, nterms, depths = parse(cfg)
		cfg.close()
		g = generator(terms, nterms, depths)

		for i in range(100):
			root = g.randTree()
			root.shapeinit(np.random.randint(2, high=9, size=random.randint(MIN_RANK, MAX_RANK)))
			createorder = _randArr(_treeSize(root))
			script = tfgen(root, createorder)
			try:
				exec(script)
			except Exception as e:
				print(script)
				raise e

if __name__ == "__main__":
	unittest.main()
