''' lAST extension allowing shape '''

import random
import numpy as np

from rast.last import node
from rast.plugin import setBuilder

# max (6 - 1) max root rank,
# for potential reduce which expands child by 1 rank
MAX_RANK = 5
MIN_RANK = 1

def _expand(shape, idx, mul):
	after = shape[idx:]
	out = list(shape[:idx])
	out.append(mul)
	out.extend(after)
	return out

def _ELEM(node):
	for arg in node.inputs:
		arg.shapeinit(node.shape)

def _REDUCE(node):
	shape = node.shape
	rank = len(shape)
	if len(node.inputs) > 1:
		limit = rank + 1
		idx = random.randint(0, rank)
		mul = random.randint(1, 9)
		node.inputs[0].shapeinit(_expand(shape, idx, mul))
		node.inputs[1].shapeinit([1], idx)
	else:
		shape = list(np.random.randint(1, high=9, size=random.randint(MIN_RANK, MAX_RANK)))
		node.inputs[0].shapeinit(shape)

def _MATMUL(node):
	assert len(node.inputs) == 2
	shape = node.shape
	if len(shape) < 2:
		shape = [shape[0], 1]
	common = random.randint(1, 9)
	if len(shape) > 2:
		beyond = shape[:-2]
	else:
		beyond = []
	node.inputs[0].shapeinit(list(beyond) + [shape[-2], common])
	node.inputs[1].shapeinit(list(beyond) + [common, shape[-1]])

SHAPER = {
	'ELEM': _ELEM,
	'REDUCE': _REDUCE,
	'MATMUL': _MATMUL
}
SHAPE_KEY = 'shape'

class shapeNode(node):
	def __init__(self, name, itypes, 
		parent=None, i=0, attr={}):
		super(shapeNode, self).__init__(name, itypes, parent, i, attr)
		if SHAPE_KEY in attr:
			self.shapeLabel = attr[SHAPE_KEY]
		else:
			self.shapeLabel = 'LEAF'
		self.shape = None
		self.scalar = None

	def __repr__(self):
		s = super(shapeNode, self).__repr__()
		if self.scalar:
			s = s + "<%d>" % self.scalar
		else:
			s = s + "[%s]" % str(self.shape)
		return s

	def shapeinit(self, shape, scalar=None):
		assert len(shape) > 0
		self.shape = shape
		self.scalar = scalar
		if self.shapeLabel in SHAPER:
			SHAPER[self.shapeLabel](self)

setBuilder(lambda name, itypes, parent, i, attr: \
	shapeNode(name, itypes, parent, i, attr))
