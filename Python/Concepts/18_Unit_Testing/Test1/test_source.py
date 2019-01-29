#! /usr/bin/env python3

''' Unit testing file for the source file '''

import unittest
import source

class TestSource(unittest.TestCase):
    ''' test class to  make the methods for testing the original methods
    All the test methods inside test class should start form the name 'test_'
    or else the method will be skipped from unit testing. This is by convention. '''

    def test_add(self):
        ''' test method for testing add function '''
        self.assertEqual(source.add(10, 5), 15)
        self.assertEqual(source.add(-1, 1), 0)
        self.assertEqual(source.add(-1, -1), -2)

    def test_sub(self):
        ''' test method for testing sub function '''
        self.assertEqual(source.sub(10, 5), 5)
        self.assertEqual(source.sub(-1, 1), -2)
        self.assertEqual(source.sub(-1, -1), 0)

    def test_mul(self):
        ''' test method for testing mul function '''
        self.assertEqual(source.mul(10, 5), 50)
        self.assertEqual(source.mul(-1, 1), -1)
        self.assertEqual(source.mul(-1, -1), 1)

    def test_div(self):
        ''' test method for testing div function '''
        self.assertEqual(source.div(10, 5), 2)
        self.assertEqual(source.div(-1, 1), -1)
        self.assertEqual(source.div(-1, -1), 1)
        self.assertEqual(source.div(5, 2), 2.5)

        # This will consider the exceptions
        # self.assertRaises(ValueError, source.div, 10, 0) --> 1st way
        # 2nd way: context manager
        with self.assertRaises(ValueError):
            source.div(10, 0)

# Alternative for python -m unittest test_source.py
if __name__ == '__main__':
    unittest.main()
