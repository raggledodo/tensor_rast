''' Serializing ast graph as a tensorflow script '''

import random
import string

from tensor_rast.shaped_node import *

TF_SCRIPT = '''import os
import random
import tensorflow as tf

LOG_RUNTIME = 'LOG_RUNTIME' in os.environ

{0}
{1}

init = tf.global_variables_initializer()
outs = []

with tf.Session() as sess:
	sess.run(init)

	if LOG_RUNTIME:
		print('adding random double scalar results')
	dscalarmap = {2}
	for label in dscalarmap:
		dscalar = dscalarmap[label]
		outs.append(('dscalar', label, dscalar))

	if LOG_RUNTIME:
		print('adding random integer scalar results')
	iscalarmap = {3}
	for label in iscalarmap:
		iscalar = iscalarmap[label]
		outs.append(('iscalar', label, iscalar))

	if LOG_RUNTIME:
		print('adding variable results')
	varmap = {4}
	for label in varmap:
		input = varmap[label]
		res = sess.run(input)
		res = res.astype(float)
		outs.append(('variable', label, res))

	if LOG_RUNTIME:
		print('adding gradient results')
	gradmap = {5}
	for label in gradmap:
		gradres = gradmap[label]
		res = sess.run(gradres)
		res = res.astype(float)
		outs.append(('gradient', label, res))

	if LOG_RUNTIME:
		print('adding output results')
	outs.append(('output', '{6}', sess.run({6}).astype(float)))
'''

class declarable:
	def __init__(self, createOrder):
		self.createOrder = createOrder
		self.leaves = []
		self.dbScalars = []
		self.intScalars = []
		self.i = 0

	def nextId(self):
		id = self.createOrder[self.i]
		self.i = self.i + 1
		return id

	def declare(self, snode, deps):
		id = self.nextId()
		if snode.name == 'scalar_double':
			if snode.scalar is not None:
				decl = str(snode.scalar)
			else:
				decl = "random.random()"
				self.dbScalars.append(id)
		elif snode.name == 'scalar_int':
			if snode.scalar is not None:
				decl = str(snode.scalar)
			else:
				decl = "random.randint(1, 9)"
				self.intScalars.append(id)
		elif snode.name == 'variable':
			decl = 'tf.Variable(tf.random_uniform([%s]))' % \
				(', '.join(list(map(lambda d: str(d), snode.shape))))
			self.leaves.append(id)
		else:
			funcname = snode.name.lower()
			if 'rmax' == funcname:
				funcname = 'reduce_max'
			elif 'rsum' == funcname:
				funcname = 'reduce_sum'
			elif 'neg' == funcname:
				funcname = 'negative'
			elif 'sub' == funcname:
				funcname = 'subtract'
			elif 'mul' == funcname:
				funcname = 'multiply'
			decl = 'tf.%s(%s)' % (funcname, ', '.join(deps))
		return id, '%s = %s' % (id, decl)

# bottom up traversal with collection
def traverse(root, func):
	collection = []
	deps = []
	if root.inputs:
		for arg in root.inputs:
			depid, coll = traverse(arg, func)
			collection.extend(coll)
			deps.append(depid)
	id, info = func(root, deps)
	collection.append(info)
	return id, collection

def tfgen(root, createOrder, script_prefix = '', script_postfix = ''):
	assert isinstance(root, shapeNode)
	decl = declarable(createOrder)
	rootID, lines = traverse(root, decl.declare)
	grads = ['grad_' + leaf for leaf in decl.leaves]
	if len(decl.leaves) > 1:
		tfGrad = '%s = tf.gradients(%s, [%s])' % \
			(', '.join(grads), rootID, ', '.join(decl.leaves))
	else:
		tfGrad = '%s = tf.gradients(%s, %s)[0]' % \
			(', '.join(grads), rootID, decl.leaves[0])

	oneOneFmt = "'{0}': {0}"
	dScalarMap = ', '.join([ oneOneFmt.format(dScalar) for dScalar in decl.dbScalars ])
	iScalarMap = ', '.join([ oneOneFmt.format(iScalar) for iScalar in decl.intScalars ])
	leafMap = ', '.join([ oneOneFmt.format(leaf) for leaf in decl.leaves ])
	gradMap = ', '.join([ "'{0}': grad_{0}".format(leaf) for leaf in decl.leaves ])

	return (script_prefix + TF_SCRIPT + script_postfix).format(
		'\n'.join(lines),
		tfGrad,
		'{' + dScalarMap + '}',
		'{' + iScalarMap + '}',
		'{' + leafMap + '}',
		'{' + gradMap + '}',
		rootID)
