import sys
import os
import pandas as pd
import unittest
import numpy as np

import loadDataExample as ld
import MaUtil

import itertools

class loadDataExampleTest(unittest.TestCase):

	def testprepareFakeDataTestSize(self):
		for i in range(1, 200, 5):
			for j in range(1, 20):
				if j + 1 >= i:
					with self.assertRaises(AssertionError):
						ld.prepareFakeData(700, 0, -5, 30, i, j)
				else:
					testdata = ld.prepareFakeData(700, 0, -5, 30, i, j)
					self.assertEqual(testdata.shape[0], i - j)

	def testprepareFakeDataTestCommonCase(self):
		# for i in range(600, 1000, 10):
		for j in range(30):
			# for k in range(5, 15, 1):
			testdata = ld.prepareFakeData(700, 0, j, 30, 75, 5)
			for l in testdata:
				for h in range(1,5):
					self.assertAlmostEqual(l[0][h] - l[0][h-1], j)








if __name__ == '__main__':
	unittest.main()
