import unittest
# import sys
# import io
# from contextlib import contextmanager

from sudoku_solver import *

unittest.TestCase.shortDescription = lambda x: None

# @contextmanager
# def captured_output():
#     new_out, new_err = io.StringIO(), io.StringIO()
#     old_out, old_err = sys.stdout, sys.stderr
#     try:
#         sys.stdout, sys.stderr = new_out, new_err
#         yield sys.stdout, sys.stderr
#     finally:
#         sys.stdout, sys.stderr = old_out, old_err


class TestSolver(unittest.TestCase):

    def test_init_exceptions(self):
        self.assertRaises(NoStringInputError, SudokuGame)
        self.assertRaises(CreateNInputStringConflictError, SudokuGame, '1', True)
        self.assertRaises(WrongInputLengthError, SudokuGame, '1')
        self.assertRaises(WrongInputTypeError, SudokuGame, 1)

    def test_init_arr(self):
        '''Tests if arr variable is initiated properly as 2D array with int and str'''

        g = SudokuGame('089005140300817006710604380043900000970000014000008730096402071400159002021700490', solve=False)
        self.assertEqual(g.arr, [['26', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '259', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['12568', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
    def test_elim(self):
        '''
        Tests if elim() method
            1. removes single str value from str entry in arr at (row, col)
            2. removes multiple str values from str entry in arr at (row, col)
            3. removes str values from str entry in arr at (row, col) if invert = True
            4. does nothing if arr[row][col] is an int
            5. allows the existence of empty string after removing all possibilities
        '''

        g = SudokuGame('089005140300817006710604380043900000970000014000008730096402071400159002021700490', solve=False)
        g.elim(0, 0, '2')
        self.assertEqual(g.arr, [['6', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '259', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['12568', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
        g.elim(1, 6, '59')
        self.assertEqual(g.arr, [['6', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '2', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['12568', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
        g.elim(3, 0, '6', invert=True)
        self.assertEqual(g.arr, [['6', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '2', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['6', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
        g.elim(0, 1, '8')
        self.assertEqual(g.arr, [['6', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '2', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['6', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
        g.elim(0, 0, '6')
        self.assertEqual(g.arr, [['', 8, 9, '23', '23', 5, 1, 4, '7'],
                                 [3, '5', '245', 8, 1, 7, '2', '25', 6], 
                                 [7, 1, '25', 6, '29', 4, 3, 8, '59'], 
                                 ['6', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['1256', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])
        
    def test_add(self):
        '''
        Tests if add() method replaces the intended place with int n.
        Also tests elimations of n from row, column, and box.
        '''

        g = SudokuGame('089005140300817006710604380043900000970000014000008730096402071400159002021700490', solve=False)
        g.add(0, 0, 2)
        self.assertEqual(g.arr, [[2, 8, 9, '3', '3', 5, 1, 4, '7'],
                                 [3, '5', '45', 8, 1, 7, '259', '25', 6], 
                                 [7, 1, '5', 6, '29', 4, 3, 8, '59'], 
                                 ['1568', 4, 3, 9, '267', '16', '2568', '256', '58'], 
                                 [9, 7, '258', '235', '236', '36', '2568', 1, 4], 
                                 ['156', '56', '25', '25', '246', 8, 7, 3, '59'], 
                                 ['58', 9, 6, 4, '38', 2, '58', 7, 1], 
                                 [4, '3', '78', 1, 5, 9, '68', '6', 2], 
                                 ['58', 2, 1, 7, '368', '36', 4, 9, '358']])

if __name__ == '__main__':
    unittest.main(verbosity=2)