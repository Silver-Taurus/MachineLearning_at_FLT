''' Unit testing file for the source file '''

import unittest
from unittest.mock import patch
from source import Employee

class TestSource(unittest.TestCase):
    ''' test class to  make the methods for testing the original methods '''

    def test_email(self):
        ''' test method for email function '''
        emp1 = Employee('Ayush', 'Kumar', 1000000)
        emp2 = Employee('Rahul', 'Sagar', 40000)

        self.assertEqual(emp1.email, 'Ayush.Kumar@email.com')
        self.assertEqual(emp2.email, 'Rahul.Sagar@email.com')

        emp1.first = 'Silver'
        emp1.last = 'Taurus'
        emp2.first = 'Ritik'

        self.assertEqual(emp1.email, 'Silver.Taurus@email.com')
        self.assertEqual(emp2.email, 'Ritik.Sagar@email.com')

    def test_fullname(self):
        ''' test method for fullname function '''
        emp1 = Employee('Ayush', 'Kumar', 1000000)
        emp2 = Employee('Rahul', 'Sagar', 40000)

        self.assertEqual(emp1.fullname, 'Ayush Kumar')
        self.assertEqual(emp2.fullname, 'Rahul Sagar')

        emp1.first = 'Silver'
        emp1.last = 'Taurus'
        emp2.first = 'Ritik'

        self.assertEqual(emp1.fullname, 'Silver Taurus')
        self.assertEqual(emp2.fullname, 'Ritik Sagar')

    def test_apply_raise(self):
        ''' test method for apply_raise function '''
        emp1 = Employee('Ayush', 'Kumar', 1000000)
        emp2 = Employee('Rahul', 'Sagar', 40000)

        emp1.apply_raise()
        emp2.apply_raise()

        self.assertEqual(emp1.pay, 1050000)
        self.assertEqual(emp2.pay, 42000)

    def test_monthly_schedule(self):
        ''' test method for monthly_schedule function '''
        with patch('source.requests.get') as mocked_get:
            mocked_get.return_value.ok = True
            mocked_get.return_value.text = 'Success'
    
            emp1 = Employee('Ayush', 'Kumar', 1000000)

            schedule = emp1.monthly_schedule('August')
            mocked_get.assert_called_with('http://company.com/Kumar/August')
            self.assertEqual(schedule, 'Success')


if __name__ == '__main__':
    unittest.main()
    