import numpy as np
from collections import Counter
from itertools import permutations, product, combinations
import operator
import random
import math
from copy import deepcopy
from enum import Enum

from pprint import pprint

from analyzer import Analysis

c_red = '\033[91m'
c_green = '\033[92m'
c_blue = '\033[94m'
c_purple = '\033[95m'
c_reset = '\033[0m'

ChainColor = Enum('ChainColor', 'ON OFF')


class SudokuGame:

    def __init__(self, arr: str = '', create=False):
        if arr == '' and not create:
            arr = 'ERROR: Needs input string for game initialization if game is not being created'
            print(arr)
            return
        elif arr != '' and create:
            arr = 'ERROR: Cannot create game and have input string to solve a game'
            print(arr)
            return

        if create:
            arr = [0] * 81
        else:
            if type(arr) == str:
                arr = [int(x) for x in arr] if len(arr) == 81 else 'ERROR: Wrong input length: needs 81 nums as str'
            else:
                arr = 'ERROR: Wrong input type: needs str'
            
            if type(arr) == str:
                print(arr)
                return

        self.arr = np.array(arr).reshape(9, 9).tolist()
        self.added = []

        self.init_arr()
        self.remaining = 81 - sum([1 if type(self.arr[row][col]) == int else 0 for row, col in product(range(9), repeat=2)])
        self.analysis = Analysis(self.get_num_psbl_nums(), init_arr=deepcopy(self.arr))

        if create:
            self.create_game()

    def __repr__(self):
        self.analysis.plot()
        return ''

    def __str__(self):
        try:
            o = ''
            for row in range(9):
                o += '-' * ((9 + 2) * 2 - 1) + '\n' if row in [3, 6] else ''
                for col in range(9):
                    o += '| ' if col in [3, 6] else ''
                    if type(self.arr[row][col]) == str:
                        o += ' ' * 2
                    else:
                        o += f'{c_red}{self.arr[row][col]}{c_reset} ' if (row, col) in self.added else f'{self.arr[row][col]} '
                o += '\n'
            return o
        except AttributeError:
            return 'Game not initialized'

    def print_psbl(self):
        try:
            width = (9 * 8 - 1) + 2
            eqline = {x: x + '=' * width + c_reset for x in (c_blue, c_purple)}
            o = eqline[c_purple] + '\n'
            for row in range(9):
                if row > 0:
                    o += '\n' + c_blue + eqline[c_blue] if row in [3, 6] else '\n' + '-' * width
                for srow in range(3):
                    o += '\n' if not (row == 0 and srow == 0) else ''
                    for col in range(9):
                        if col > 0:
                            o += c_blue + '|' + c_reset if col in [3, 6] else '|'
                        o += c_blue + '|' + c_reset if col in [3, 6] else ''
                        if type((v := self.arr[row][col])) == str:
                            o += ' ' + ''.join([str(psbl_val) + ' ' if (psbl_val := str(srow * 3 + scol + 1)) in v else ' ' * 2 for scol in range(3)])
                        else:
                            color = c_red if (row, col) in self.added else c_green
                            o += ' ' * 3 + f'{color}{v}{c_reset}' + ' ' * 3 if srow == 1 else ' ' * 7
            o += '\n' + eqline[c_purple]
            print(o)
        except AttributeError:
            return 'Game not initialized'

    def analysis_func_append(self, func):
        self.analysis.func_append(func, self.get_num_psbl_nums(), self.get_num_psbl_nodes())
        self.analysis.steps.append([])

    def analysis_step_append(self, lst):
        self.analysis.steps[-1].append(lst)

    def analysis_append(self, func, lst):
        self.analysis_step_append(lst)
        self.analysis_func_append(func)

    def init_arr(self):
        for row, col in product(range(9), repeat=2):
            if self.arr[row][col] == 0:
                self.arr[row][col] = self.CHRSEQ
                for idx in range(9):
                    vr, vc = [str(v) if type(v) == int else '' for v in (self.arr[idx][col], self.arr[row][idx])]
                    self.arr[row][col] = self.arr[row][col].replace(vr, '').replace(vc, '')
                ltrow, ltcol = [(x // 3) * 3 for x in (row, col)]
                for brow, bcol in product([0, 1, 2], repeat=2):
                    if type((v := self.arr[ltrow + brow][ltcol + bcol])) == int:
                        self.arr[row][col] = self.arr[row][col].replace(str(v), '')

    def create_game(self):
        for row in range(9):
            for col in range(9):
                self.solve_singles()
                v = self.arr[row][col]
                if type(v) == str:
                    l = [int(x) for x in v]
                    rand = random.choice(l)
                    self.add(row, col, rand)
                    self.print_psbl()

    def get_num_psbl_nums(self):
        return sum([len(v) for row, col in product(range(9), repeat=2) if type((v := self.arr[row][col])) == str])

    def get_num_psbl_nodes(self):
        return int(math.prod([len(v) for row, col in product(range(9), repeat=2) if type((v := self.arr[row][col])) == str]))

    def add(self, row, col, n: int, chain = False):
        self.added.append((row, col))
        self.remaining -= 1

        elim_cnt = 0
        elim_lst = []

        elim = self.elim(row, col, str(n), invert=True)
        elim_cnt += elim[0]
        elim_lst += elim[1]

        letters = ['a', 'ae']
        if chain:
            letters = ['c'+x for x in letters]

        if elim_cnt == 0:
            elim_lst += [(letters[0], row, col, n)]
        else:
            elim_lst += [(letters[1], row, col, {'a': n, 'e': elim_lst[-1][3]})]
            del elim_lst[-2]

        # Since each strategy has an internal while loop, the following algorithm of using either-or chains
        # to add candidates is not necessary

        if False:
            single_chains = self.get_either_or_chains()
            add_lst = []
            for chain in single_chains[str(n)]:
                chain_tups = [(t[0], t[1]) for t in chain]
                t = (row, col)
                if t in chain_tups:
                    color = chain[chain_tups.index(t)][2]
                    remove_color = ChainColor.OFF if color == ChainColor.ON else ChainColor.ON
                    for tup in chain:
                        if tup[2] == remove_color:
                            elim = self.elim(tup[0], tup[1], str(n), chain=True)
                            elim_cnt += elim[0]
                            elim_lst += elim[1]
                        elif tup[2] == color:
                            add_lst.append((tup[0], tup[1], n))
                    break

            for t in add_lst:
                elim = self.add(*t, chain=True)
                elim_cnt += elim[0]
                self.analysis_step_append(elim[1])

        self.arr[row][col] = n
        elim_cnt += 1

        for idx in range(9):
            if idx != col and type(self.arr[row][idx]) == str:
                elim = self.elim(row, idx, str(n))
                elim_cnt += elim[0]
                elim_lst += elim[1]
            if idx != row and type(self.arr[idx][col]) == str:
                elim = self.elim(idx, col, str(n))
                elim_cnt += elim[0]
                elim_lst += elim[1]
        ltrow, ltcol = [(x // 3) * 3 for x in [row, col]]
        for brow, bcol in self.BSEQ:
            if (ltrow + brow, ltcol + bcol) == (row, col):
                continue
            elim = self.elim(ltrow + brow, ltcol + bcol, str(n)) if type(self.arr[ltrow + brow][ltcol + bcol]) == str else (0, [])
            elim_cnt += elim[0]
            elim_lst += elim[1]
        
        return elim_cnt, elim_lst

    def elim(self, row, col, lst: str | list[str], invert=False, chain=False):
        lst = ''.join(set(self.CHRSEQ) - set(lst if type(lst) == str else ''.join(lst))) if invert else lst
        elim_lst = []
        for char in lst:
            if type((v := self.arr[row][col])) == str and char in v:
                self.arr[row][col] = v.replace(char, '')
                elim_lst.append(int(char))
        ret_char = 'ce' if chain else 'e'
        return len(elim_lst), [(ret_char, row, col, elim_lst) if len(elim_lst) > 0 else None]

    def solve_singles(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            for row, col in product(range(9), repeat=2):
                if type((v := self.arr[row][col])) == str and len(v) == 1:
                    elim = self.add(row, col, int(v))
                    elim_cnt += elim[0]
                    self.analysis_step_append(elim[1])
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.solve_singles)

    def solve_hidden_singles(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            for row in range(9):
                count = sum([Counter(v) for col in range(9) if type((v := self.arr[row][col])) == str], Counter())
                l = [val for val, cnt in count.items() if cnt == 1]
                for col, char in product(range(9), l):
                    if type((v := self.arr[row][col])) == str and char in v:
                        elim = self.add(row, col, int(char))
                        elim_cnt += elim[0]
                        self.analysis_step_append(elim[1])
            for col in range(9):
                count = sum([Counter(v) for row in range(9) if type((v := self.arr[row][col])) == str], Counter())
                l = [val for val, cnt in count.items() if cnt == 1]
                for row, char in product(range(9), l):
                    if type((v := self.arr[row][col])) == str and char in v:
                        elim = self.add(row, col, int(char))
                        elim_cnt += elim[0]
                        self.analysis_step_append(elim[1])
            for ltrow, ltcol in product([0, 3, 6], repeat=2):
                count = sum(
                    [Counter(v) for brow, bcol in product([0, 1, 2], repeat=2) if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str], Counter()
                )
                l = [val for val, cnt in count.items() if cnt == 1]
                for brow, bcol, char in product([0, 1, 2], [0, 1, 2], l):
                    if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str and char in v:
                        elim = self.add(ltrow + brow, ltcol + bcol, int(char))
                        elim_cnt += elim[0]
                        self.analysis_step_append(elim[1])
            if self.get_num_psbl_nums() == old:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.solve_hidden_singles)

    def elim_naked_pairs(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            for row in range(9):
                psbl, pairs = [], []
                for col in range(9):
                    if type((v := self.arr[row][col])) == str and len(v) == 2:
                        pairs.append(v) if v in psbl else psbl.append(v)
                for p, col in product(pairs, range(9)):
                    if type((v := self.arr[row][col])) == str and v != p:
                        elim = self.elim(row, col, p)
                        elim_cnt += elim[0]
                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in range(9):
                psbl, pairs = [], []
                for row in range(9):
                    if type((v := self.arr[row][col])) == str and len(v) == 2:
                        pairs.append(v) if v in psbl else psbl.append(v)
                for p, row in product(pairs, range(9)):
                    if type((v := self.arr[row][col])) == str and v != p:
                        elim = self.elim(row, col, p)
                        elim_cnt += elim[0]
                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in product([0, 3, 6], repeat=2):
                psbl, pairs = [], []
                for brow, bcol in product([0, 1, 2], repeat=2):
                    if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str and len(v) == 2:
                        pairs.append(v) if v in psbl else psbl.append(v)
                for p, brow, bcol in product(pairs, [0, 1, 2], [0, 1, 2]):
                    if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str and v != p:
                        elim = self.elim(ltrow + brow, ltcol + bcol, p)
                        elim_cnt += elim[0]
                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_naked_pairs)

    def elim_naked_triples(self):
        elim_cnt = 0
        while True:
            old_psbl = self.get_num_psbl_nums()
            elim_lst = []
            for row in range(9):
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, col) for col in range(9) if type((v := self.arr[row][col])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3]]
                    for comb in combinations(strings, 3):
                        if len(all_str) > 3 and len((s := set.union(*[set(c[0]) for c in comb]))) == 3:
                            for col in range(9):
                                if col not in [c[1] for c in comb]:
                                    elim = self.elim(row, col, ''.join(s))
                                    elim_cnt += elim[0]
                                    elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in range(9):
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, row) for row in range(9) if type((v := self.arr[row][col])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3]]
                    for comb in combinations(strings, 3):
                        if len(all_str) > 3 and len((s := set.union(*[set(c[0]) for c in comb]))) == 3:
                            for row in range(9):
                                if row not in [c[1] for c in comb]:
                                    elim = self.elim(row, col, ''.join(s))
                                    elim_cnt += elim[0]
                                    elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in product([0, 3, 6], repeat=2):
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, brow, bcol) for brow, bcol in product([0, 1, 2], repeat=2) if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3]]
                    for comb in combinations(strings, 3):
                        if len(all_str) > 3 and len((s := set.union(*[set(c[0]) for c in comb]))) == 3:
                            for brow, bcol in product([0, 1, 2], repeat=2):
                                if (brow, bcol) not in [(c[1], c[2]) for c in comb]:
                                    elim = self.elim(ltrow + brow, ltcol + bcol, ''.join(s))
                                    elim_cnt += elim[0]
                                    elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old_psbl:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_naked_triples)
    
    def elim_naked_quads(self):
        elim_cnt = 0
        while True:
            old_psbl = self.get_num_psbl_nums()
            elim_lst = []
            for row in self.RCSEQ:
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3, 4]]
                    if len(all_str) > 4:
                        for comb in combinations(strings, 4):
                            if len((s := set.union(*[set(c[0]) for c in comb]))) == 4:
                                for col in range(9):
                                    if col not in [c[1] for c in comb]:
                                        elim = self.elim(row, col, ''.join(s))
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in self.RCSEQ:
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3, 4]]
                    if len(all_str) > 4:
                        for comb in combinations(strings, 4):
                            if len((s := set.union(*[set(c[0]) for c in comb]))) == 4:
                                for row in range(9):
                                    if row not in [c[1] for c in comb]:
                                        elim = self.elim(row, col, ''.join(s))
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in self.LTSEQ:
                while True:
                    old = self.get_num_psbl_nums()
                    all_str = [(v, brow, bcol) for brow, bcol in self.BSEQ if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                    strings = [v for v in all_str if len(v[0]) in [2, 3, 4]]
                    if len(all_str) > 4:
                        for comb in combinations(strings, 4):
                            if len((s := set.union(*[set(c[0]) for c in comb]))) == 4:
                                for brow, bcol in self.BSEQ:
                                    if (brow, bcol) not in [(c[1], c[2]) for c in comb]:
                                        elim = self.elim(ltrow + brow, ltcol + bcol, ''.join(s))
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                    if self.get_num_psbl_nums() == old:
                        break
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old_psbl:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_naked_quads)

    def elim_hidden_pairs(self):
        def _elim_hidden_pairs(lst):
            count = sum([Counter(v[0]) for v in lst], Counter())
            l = [val for val, cnt in count.items() if cnt == 2]
            cnt = []
            if len(l) % 2 == 0:
                comb = [''.join(t) for t in combinations(sorted(l), 2)]
                cnt = [(0, list(), '')] * len(comb)
                for string, c in product(lst, comb):
                    v = cnt[comb.index(c)]
                    if all([char in string[0] for char in c]):
                        t_add = (1, [lst[lst.index(string)][1]], comb[comb.index(c)] if v[2] == '' else '')
                        v = tuple(map(operator.add, v, t_add))
                    cnt[comb.index(c)] = v
            return cnt

        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            for row in range(9):
                lst = [(v, col) for col in range(9) if type((v := self.arr[row][col])) == str]
                count = _elim_hidden_pairs(lst)
                for t in count:
                    for col in t[1]:
                        if t[0] == 2:
                            elim = self.elim(row, col, t[2], invert=True)
                            elim_cnt += elim[0]
                            elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in range(9):
                lst = [(v, row) for row in range(9) if type((v := self.arr[row][col])) == str]
                count = _elim_hidden_pairs(lst)
                for t in count:
                    for row in t[1]:
                        if t[0] == 2:
                            elim = self.elim(row, col, t[2], invert=True)
                            elim_cnt += elim[0]
                            elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in product([0, 3, 6], repeat=2):
                lst = [(v, brow * 3 + bcol) for brow, bcol in product([0, 1, 2], repeat=2) if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                count = _elim_hidden_pairs(lst)
                for t in count:
                    for idx in t[1]:
                        if t[0] == 2:
                            elim = self.elim(ltrow + ((idx - idx % 3) // 3), ltcol + (idx % 3), t[2], invert=True)
                            elim_cnt += elim[0]
                            elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break

        if elim_cnt > 0:
            self.analysis_func_append(self.elim_hidden_pairs)

    def elim_hidden_triples(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            for row in self.RCSEQ:
                all_str = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, col_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 3):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 3):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                elim = self.elim(row, col_lst[val_lst.index(s)], ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in self.RCSEQ:
                all_str = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, row_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 3):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 3):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                elim = self.elim(row_lst[val_lst.index(s)], col, ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in self.LTSEQ:
                all_str = [(v, brow, bcol) for brow, bcol in self.BSEQ if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                val_lst, b_lst = [t[0] for t in all_str], [(t[1], t[2]) for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 3):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 3):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                brow, bcol = b_lst[val_lst.index(s)]
                                elim = self.elim(ltrow + brow, ltcol + bcol, ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_hidden_triples)

    def elim_hidden_quads(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            for row in self.RCSEQ:
                all_str = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, col_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3, 4]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 4):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 4):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                elim = self.elim(row, col_lst[val_lst.index(s)], ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in self.RCSEQ:
                all_str = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, row_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3, 4]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 4):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 4):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                elim = self.elim(row_lst[val_lst.index(s)], col, ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in self.LTSEQ:
                all_str = [(v, brow, bcol) for brow, bcol in self.BSEQ if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                val_lst, b_lst = [t[0] for t in all_str], [(t[1], t[2]) for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_set = set([v for v in all_str_cnt.items() if v[1] in [2, 3, 4]])
                filtered_keys = [v[0] for v in filtered_set]
                for comb in combinations(val_lst, 4):
                    comb_cnt = sum([Counter(t) for t in comb], Counter())
                    filtered_subset = set([v for v in comb_cnt.items() if v[0] in filtered_keys])
                    for fcomb in combinations(filtered_subset, 4):
                        if set(fcomb) <= filtered_set:
                            for s in comb:
                                brow, bcol = b_lst[val_lst.index(s)]
                                elim = self.elim(ltrow + brow, ltcol + bcol, ''.join([v[0] for v in fcomb]), invert=True)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_hidden_quads)

    def elim_intersection_removal(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            for row in self.RCSEQ:
                all_str = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str]
                all_str_chr = []
                for t in all_str:
                    for c in t[0]:
                        all_str_chr.append((c, t[1]))
                all_str_dict = {}
                for t in all_str_chr:
                    if t[0] in all_str_dict:
                        all_str_dict[t[0]].append(t[1])
                    else:
                        all_str_dict[t[0]] = [t[1]]
                ltrow, brow = row // 3 * 3, row % 3
                for char in all_str_dict:
                    cols = all_str_dict[char]
                    btups = [(brow, col % 3) for col in cols]
                    can_elim = False
                    if min(cols) >= 0 and max(cols) <= 2:
                        ltcol = 0
                        can_elim = True
                    elif min(cols) >= 3 and max(cols) <= 5:
                        ltcol = 3
                        can_elim = True
                    elif min(cols) >= 6 and max(cols) <= 8:
                        ltcol = 6
                        can_elim = True
                    if can_elim:
                        for br, bc in self.BSEQ:
                            if type((v := self.arr[ltrow + br][ltcol + bc])) == str and (br, bc) not in btups:
                                elim = self.elim(ltrow + br, ltcol + bc, char)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for col in self.RCSEQ:
                all_str = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str]
                all_str_chr = []
                for t in all_str:
                    for c in t[0]:
                        all_str_chr.append((c, t[1]))
                all_str_dict = {}
                for t in all_str_chr:
                    if t[0] in all_str_dict:
                        all_str_dict[t[0]].append(t[1])
                    else:
                        all_str_dict[t[0]] = [t[1]]
                ltcol, bcol = col // 3 * 3, col % 3
                for char in all_str_dict:
                    rows = all_str_dict[char]
                    btups = [(row % 3, bcol) for row in rows]
                    can_elim = False
                    if min(rows) >= 0 and max(rows) <= 2:
                        ltrow = 0
                        can_elim = True
                    elif min(rows) >= 3 and max(rows) <= 5:
                        ltrow = 3
                        can_elim = True
                    elif min(rows) >= 6 and max(rows) <= 8:
                        ltrow = 6
                        can_elim = True
                    if can_elim:
                        for br, bc in self.BSEQ:
                            if type((v := self.arr[ltrow + br][ltcol + bc])) == str and (br, bc) not in btups:
                                elim = self.elim(ltrow + br, ltcol + bc, char)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            for ltrow, ltcol in self.LTSEQ:
                all_str = [(v, brow, bcol) for brow, bcol in self.BSEQ if type((v := self.arr[ltrow + brow][ltcol + bcol])) == str]
                all_str_chr = []
                for t in all_str:
                    for c in t[0]:
                        all_str_chr.append((c, t[1], t[2]))
                all_str_dict = {}
                for t in all_str_chr:
                    if t[0] in all_str_dict:
                        all_str_dict[t[0]][0].add(t[1])
                        all_str_dict[t[0]][1].add(t[2])
                    else:
                        all_str_dict[t[0]] = [set([t[1]]), set([t[2]])]
                for char in all_str_dict:
                    if len(all_str_dict[char][0]) == 1:
                        brow = list(all_str_dict[char][0])[0]
                        row = ltrow + brow
                        found_lst = [(ltrow + brow, ltcol + bcol) for bcol in all_str_dict[char][1]]
                        for col in self.RCSEQ:
                            if (row, col) not in found_lst:
                                elim = self.elim(row, col, char)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
                    elif len(all_str_dict[char][1]) == 1:
                        bcol = list(all_str_dict[char][1])[0]
                        col = ltcol + bcol
                        found_lst = [(ltrow + brow, ltcol + bcol) for brow in all_str_dict[char][0]]
                        for row in self.RCSEQ:
                            if (row, col) not in found_lst:
                                elim = self.elim(row, col, char)
                                elim_cnt += elim[0]
                                elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_intersection_removal)

    def elim_x_wing(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            all_tup_lst_rows = []
            all_tup_lst = []
            for row in self.RCSEQ:
                all_str = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, col_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_keys = [k for k, v in all_str_cnt.items() if v == 2]
                tup_lst = []
                for k in filtered_keys:
                    cols = []
                    for i in range(len(val_lst)):
                        if k in val_lst[i]:
                            cols.append(col_lst[i])
                    tup_lst.append((k, *cols))
                for t in tup_lst:
                    if t in all_tup_lst:
                        idx = all_tup_lst.index(t)
                        rows = [all_tup_lst_rows[idx], row]
                        for r in self.RCSEQ:
                            if r not in rows:
                                for c in t[1:]:
                                    elim = self.elim(r, c, t[0])
                                    elim_cnt += elim[0]
                                    elim_lst += elim[1]
                    else:
                        all_tup_lst_rows.append(row)
                        all_tup_lst.append(t)
            self.analysis_step_append(elim_lst)
            elim_lst = []
            all_tup_lst_cols = []
            all_tup_lst = []
            for col in self.RCSEQ:
                all_str = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str]
                val_lst, col_lst = [t[0] for t in all_str], [t[1] for t in all_str]
                all_str_cnt = sum([Counter(t) for t in val_lst], Counter())
                filtered_keys = [k for k, v in all_str_cnt.items() if v == 2]
                tup_lst = []
                for k in filtered_keys:
                    rows = []
                    for i in range(len(val_lst)):
                        if k in val_lst[i]:
                            rows.append(col_lst[i])
                    tup_lst.append((k, *rows))
                for t in tup_lst:
                    if t in all_tup_lst:
                        idx = all_tup_lst.index(t)
                        cols = [all_tup_lst_cols[idx], col]
                        for c in self.RCSEQ:
                            if c not in cols:
                                for r in t[1:]:
                                    elim = self.elim(r, c, t[0])
                                    elim_cnt += elim[0]
                                    elim_lst += elim[1]
                    else:
                        all_tup_lst_cols.append(col)
                        all_tup_lst.append(t)
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_x_wing)

    def get_either_or_chains(self):
        all_str = [(v, row, col) for row, col in product(self.RCSEQ, repeat=2) if type((v := self.arr[row][col])) == str]
        val_lst, row_lst, col_lst = [t[0] for t in all_str], [t[1] for t in all_str], [t[2] for t in all_str]
        d = {}
        for c in self.CHRSEQ:
            l = [(t[1], t[2]) for t in all_str if c in t[0]]
            rl = [x[0] for x in l]
            cl = [x[1] for x in l]
            rl_cnt = Counter(rl)
            cl_cnt = Counter(cl)
            doubles = []
            for i in self.RCSEQ:
                if rl_cnt[i] == 2: # two psbl in same row
                    doubles += [[(i, cl[j]) for j in range(len(rl)) if rl[j] == i]]
                if cl_cnt[i] == 2: # two psbl in same col
                    doubles += [[(rl[j], i) for j in range(len(cl)) if cl[j] == i]]
            bl = []
            for ltrow, ltcol in self.LTSEQ:
                lst = []
                for brow, bcol in self.BSEQ:
                    row, col = ltrow + brow, ltcol + bcol
                    if (row, col) in l:
                        lst += [(row, col)]
                if len(lst) == 2:
                    doubles += [lst]

            colored_chains = []

            while len(doubles) > 0:
                i = 0
                chain = []
                tups = []
                while i < len(doubles):
                    sl = doubles.pop(0)
                    if len(tups) > 0 and sl[0] not in tups and sl[1] not in tups:
                        doubles.append(sl)
                        i += 1
                    elif sl[0] in tups:
                        idx = tups.index(sl[0])
                        color = ChainColor.OFF if chain[idx][2] == ChainColor.ON else ChainColor.ON
                        chain += [(*sl[1], color)]
                        tups += [sl[1]]
                        i = 0
                    elif sl[1] in tups:
                        idx = tups.index(sl[1])
                        color = ChainColor.OFF if chain[idx][2] == ChainColor.ON else ChainColor.ON
                        chain += [(*sl[0], color)]
                        tups += [sl[0]]
                        i = 0
                    else:
                        chain += [(*sl[0], ChainColor.ON), (*sl[1], ChainColor.OFF)]
                        tups += [sl[0], sl[1]]
                        i = 0
                colored_chains.append(chain)
            d[c] = [list(set(x)) for x in colored_chains]
        return d

    def elim_single_chains(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            chains = self.get_either_or_chains()
            for char in self.CHRSEQ:
                for chain in chains[char]:
                    rows_on, rows_off = [], []
                    cols_on, cols_off = [], []
                    lt_on, lt_off = [], []
                    b_on, b_off = [], []
                    for t in chain:
                        ltrow, ltcol = [x//3*3 for x in t[:2]]
                        brow, bcol = [x%3 for x in t[:2]]
                        if t[2] == ChainColor.ON:
                            rows_on.append(t[0])
                            cols_on.append(t[1])
                            lt_on.append((ltrow, ltcol))
                            b_on.append((brow, bcol))
                        elif t[2] == ChainColor.OFF:
                            rows_off.append(t[0])
                            cols_off.append(t[1])
                            lt_off.append((ltrow, ltcol))
                            b_off.append((brow, bcol))
                    all_lst = [rows_on, rows_off, cols_on, cols_off, lt_on, lt_off, b_on, b_off]
                    for i, lst in enumerate(all_lst[:-2]):
                        cnt_2 = [k for k, v in Counter(lst).items() if v == 2]
                        if len(cnt_2) > 0:
                            if i in [0, 1]:
                                for row in cnt_2:
                                    l = [all_lst[i+2][k] for k,v in enumerate(lst) if v == row]
                                    for col in l:
                                        elim = self.elim(row, col, char)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                            elif i in [2, 3]:
                                for col in cnt_2:
                                    l = [all_lst[i-2][k] for k,v in enumerate(lst) if v == col]
                                    for row in l:
                                        elim = self.elim(row, col, char)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                            elif i in [4, 5]:
                                for lttup in cnt_2:
                                    l = [all_lst[i+2][k] for k,v in enumerate(lst) if v == lttup]
                                    for btup in l:
                                        elim = self.elim(lttup[0] + btup[0], lttup[1] + btup[1], char)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            elim_lst = []
            chains = self.get_either_or_chains()
            for char in self.CHRSEQ:
                for chain in chains[char]:
                    for i in range(len(chain)):
                        for j in range(i+1, len(chain)):
                            ri, ci, clri = chain[i]
                            rj, cj, clrj = chain[j]
                            if clri != clrj:
                                lri, lci = [x//3*3 for x in [ri, ci]]
                                bri, bci = [x%3 for x in [ri, ci]]
                                lrj, lcj = [x//3*3 for x in [rj, cj]]
                                brj, bcj = [x%3 for x in [rj, cj]]
                                
                                check_i = [(ri, col) for col in self.RCSEQ if col != ci]
                                check_i += [(row, ci) for row in self.RCSEQ if row != ri]
                                check_i += [(lri+brow, lci+bcol) for brow, bcol in self.BSEQ if (brow, bcol) != (bri, bci)]
                                check_i = set(check_i)

                                check_j = [(rj, col) for col in self.RCSEQ if col != cj]
                                check_j += [(row, cj) for row in self.RCSEQ if row != rj]
                                check_j += [(lrj+brow, lcj+bcol) for brow, bcol in self.BSEQ if (brow, bcol) != (brj, bcj)]
                                check_j = set(check_j)

                                intersect = check_i.intersection(check_j)
                                for row, col in intersect:
                                    if type((v := self.arr[row][col])) == str and char in v:
                                        elim = self.elim(row, col, char)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_single_chains)

    def elim_y_wing(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            all_2tups = [(row, col) for row, col in product(self.RCSEQ, repeat=2) if type((v := self.arr[row][col])) == str and len(v) == 2]
            for row, col in all_2tups:
                vo = self.arr[row][col]
                ltrow, ltcol = [x//3*3 for x in [row, col]]
                brow, bcol = [x%3 for x in [row, col]]
                infl = [(x, col) for x in self.RCSEQ if x != row and (x, col) in all_2tups]
                infl += [(row, x) for x in self.RCSEQ if x != col and (row, x) in all_2tups]
                infl += [(ltrow+br, ltcol+bc) for br,bc in self.BSEQ if (br, bc) != (brow, bcol) and (ltrow+br, ltcol+bc) in all_2tups]
                infl = list(set(infl))
                for i in range(len(infl)):
                    for j in range(i+1, len(infl)):
                        ri, ci = infl[i]
                        rj, cj = infl[j]
                        vi = self.arr[ri][ci]
                        vj = self.arr[rj][cj]
                        if vi != vj and any(x in vi for x in vj):
                            c_val = [x for x in vj if x in vi][0]
                            a_val_lst = [x for x in vi if x in vo and x != c_val]
                            b_val_lst = [x for x in vj if x in vo and x != c_val]
                            if len(a_val_lst) > 0 and len(b_val_lst) > 0:
                                lri, lci = [x//3*3 for x in [ri, ci]]
                                bri, bci = [x%3 for x in [ri, ci]]
                                lrj, lcj = [x//3*3 for x in [rj, cj]]
                                brj, bcj = [x%3 for x in [rj, cj]]

                                check_i = [(ri, c) for c in self.RCSEQ if (ri, c) not in [(ri, ci), (row, col)]]
                                check_i += [(r, ci) for r in self.RCSEQ if (r, ci) not in [(ri, ci), (row, col)]]
                                check_i += [(lri+br, lci+bc) for br,bc in self.BSEQ if (br, bc) != (bri, bci) and (lri, lci, br, bc) != (ltrow, ltcol, brow, bcol)]
                                check_i = set(check_i)

                                check_j = [(rj, c) for c in self.RCSEQ if (rj, c) not in [(rj, cj), (row, col)]]
                                check_j += [(r, cj) for r in self.RCSEQ if (r, cj) not in [(rj, cj), (row, col)]]
                                check_j += [(lrj+br, lcj+bc) for br,bc in self.BSEQ if (br, bc) != (brj, bcj) and (lrj, lcj, br, bc) != (ltrow, ltcol, brow, bcol)]
                                check_j = set(check_j)

                                intersect = check_i.intersection(check_j)
                                for ins_r, ins_c in intersect:
                                    if type((v := self.arr[ins_r][ins_c])) == str and c_val in v:
                                        elim = self.elim(ins_r, ins_c, c_val)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_y_wing)

    def elim_swordfish(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            for char in self.CHRSEQ:
                elim_lst = []
                all_str_lst = []
                all_row_lst = []
                for row in self.RCSEQ:
                    str_char = [(v, col) for col in self.RCSEQ if type((v := self.arr[row][col])) == str and char in v]
                    if len(str_char) > 3:
                        str_char = []
                    str_lst, col_lst = [x[0] for x in str_char], [x[1] for x in str_char]
                    all_str_lst.append(str_lst)
                    all_row_lst.append(col_lst)
                all_row_lst_comb = [list(combinations(x, 2)) for x in all_row_lst]
                found = False
                for ri, rj, rk in product(range(len(all_row_lst_comb)), repeat=3):
                    if found:
                        break
                    if len(set([ri, rj, rk])) != 3:
                        continue
                    cli = all_row_lst_comb[ri]
                    clj = all_row_lst_comb[rj]
                    clk = all_row_lst_comb[rk]
                    for ti, tj, tk in product(cli, clj, clk):
                        if found:
                            break
                        c_set = set(ti) | set(tj) | set(tk)
                        if len(c_set) == 3 and len(set([ti, tj, tk])) == 3:
                            mci = list(c_set - set(ti))[0]
                            mcj = list(c_set - set(tj))[0]
                            mck = list(c_set - set(tk))[0]
                            mci_cond = (len((tmp := all_row_lst[ri])) == 3 and mci in tmp) or len(tmp) == 2
                            mcj_cond = (len((tmp := all_row_lst[rj])) == 3 and mcj in tmp) or len(tmp) == 2
                            mck_cond = (len((tmp := all_row_lst[rk])) == 3 and mck in tmp) or len(tmp) == 2
                            if mci_cond and mcj_cond and mck_cond:
                                for col in c_set:
                                    for row in self.RCSEQ:
                                        if row in [ri, rj, rk]:
                                            continue
                                        if type((v := self.arr[row][col])) == str and char in v:
                                            found = True
                                            elim = self.elim(row, col, char)
                                            elim_cnt += elim[0]
                                            elim_lst += elim[1]
                self.analysis_step_append(elim_lst)
                elim_lst = []
                all_str_lst = []
                all_col_lst = []
                for col in self.RCSEQ:
                    str_char = [(v, row) for row in self.RCSEQ if type((v := self.arr[row][col])) == str and char in v]
                    if len(str_char) not in [2, 3]:
                        str_char = []
                    str_lst, row_lst = [x[0] for x in str_char], [x[1] for x in str_char]
                    all_str_lst.append(str_lst)
                    all_col_lst.append(row_lst)
                all_col_lst_comb = [list(combinations(x, 2)) for x in all_col_lst]
                found = False
                for ci, cj, ck in product(range(len(all_col_lst_comb)), repeat=3):
                    if found:
                        break
                    if len(set([ci, cj, ck])) != 3:
                        continue
                    rli = all_col_lst_comb[ci]
                    rlj = all_col_lst_comb[cj]
                    rlk = all_col_lst_comb[ck]
                    for ti, tj, tk in product(rli, rlj, rlk):
                        if found:
                            break
                        r_set = set(ti) | set(tj) | set(tk)
                        if len(r_set) == 3 and len(set([ti, tj, tk])) == 3:
                            mri = list(r_set - set(ti))[0]
                            mrj = list(r_set - set(tj))[0]
                            mrk = list(r_set - set(tk))[0]
                            mri_cond = (len((tmp := all_col_lst[ci])) == 3 and mri in tmp) or len(tmp) == 2
                            mrj_cond = (len((tmp := all_col_lst[cj])) == 3 and mrj in tmp) or len(tmp) == 2
                            mrk_cond = (len((tmp := all_col_lst[ck])) == 3 and mrk in tmp) or len(tmp) == 2
                            if mri_cond and mrj_cond and mrk_cond:
                                for row in r_set:
                                    for col in self.RCSEQ:
                                        if col in [ci, cj, ck]:
                                            continue
                                        if type((v := self.arr[row][col])) == str and char in v:
                                            found = True
                                            elim = self.elim(row, col, char)
                                            elim_cnt += elim[0]
                                            elim_lst += elim[1]
                self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_swordfish)

    def elim_xyz_wing(self):
        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            elim_lst = []
            all_3tups = [(row, col) for row, col in product(self.RCSEQ, repeat=2) if type((v := self.arr[row][col])) == str and len(v) == 3]
            all_2tups = [(row, col) for row, col in product(self.RCSEQ, repeat=2) if type((v := self.arr[row][col])) == str and len(v) == 2]
            for row, col in all_3tups:
                vo = self.arr[row][col]
                ltrow, ltcol = [x//3*3 for x in [row, col]]
                brow, bcol = [x%3 for x in [row, col]]
                infl = [(x, col) for x in self.RCSEQ if (x, col) in all_2tups]
                infl += [(row, x) for x in self.RCSEQ if (row, x) in all_2tups]
                infl += [(ltrow+br, ltcol+bc) for br,bc in self.BSEQ if (ltrow+br, ltcol+bc) in all_2tups]
                infl = list(set(infl))

                check_o = [(row, c) for c in self.RCSEQ if c != col]
                check_o += [(r, col) for r in self.RCSEQ if r != row]
                check_o += [(ltrow+br, ltcol+bc) for br,bc in self.BSEQ if (br, bc) != (brow, bcol)]
                check_o = set(check_o)

                for i in range(len(infl)):
                    for j in range(i+1, len(infl)):
                        ri, ci = infl[i]
                        rj, cj = infl[j]
                        vi = self.arr[ri][ci]
                        vj = self.arr[rj][cj]
                        if vi != vj and any(x in vi for x in vj):
                            c_val = [x for x in vj if x in vi][0]
                            a_val_lst = [x for x in vi if x in vo and x != c_val]
                            b_val_lst = [x for x in vj if x in vo and x != c_val]
                            if c_val in vo and len(a_val_lst) > 0 and len(b_val_lst) > 0:
                                lri, lci = [x//3*3 for x in [ri, ci]]
                                bri, bci = [x%3 for x in [ri, ci]]
                                lrj, lcj = [x//3*3 for x in [rj, cj]]
                                brj, bcj = [x%3 for x in [rj, cj]]

                                check_i = [(ri, c) for c in self.RCSEQ if (ri, c) not in [(ri, ci), (row, col)]]
                                check_i += [(r, ci) for r in self.RCSEQ if (r, ci) not in [(ri, ci), (row, col)]]
                                check_i += [(lri+br, lci+bc) for br,bc in self.BSEQ if (br, bc) != (bri, bci) and (lri, lci, br, bc) != (ltrow, ltcol, brow, bcol)]
                                check_i = set(check_i)

                                check_j = [(rj, c) for c in self.RCSEQ if (rj, c) not in [(rj, cj), (row, col)]]
                                check_j += [(r, cj) for r in self.RCSEQ if (r, cj) not in [(rj, cj), (row, col)]]
                                check_j += [(lrj+br, lcj+bc) for br,bc in self.BSEQ if (br, bc) != (brj, bcj) and (lrj, lcj, br, bc) != (ltrow, ltcol, brow, bcol)]
                                check_j = set(check_j)

                                intersect = check_i.intersection(check_j).intersection(check_o)
                                for ins_r, ins_c in intersect:
                                    if type((v := self.arr[ins_r][ins_c])) == str and c_val in v:
                                        elim = self.elim(ins_r, ins_c, c_val)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
            self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_xyz_wing)

    def elim_x_cycles(self):
        def _same_box(rc1, rc2):
            r1, c1 = rc1
            r2, c2 = rc2
            return r1//3 == r2//3 and c1//3 == c2//3

        def _lt_convert(rc):
            return tuple([x//3*3 for x in rc])

        def _print_chains(chains):
            for chain in chains:
                o = f'{chain[0][0]}'
                for t in chain:
                    o += f' - {t[1]}'
                print(o)

        elim_cnt = 0
        while True:
            old = self.get_num_psbl_nums()
            for char in self.CHRSEQ:
                elim_lst = []
                all_rc = [(row, col) for row, col in product(self.RCSEQ, repeat=2) if type((v := self.arr[row][col])) == str and char in v]
                row_cnt = Counter([x[0] for x in all_rc])
                col_cnt = Counter([x[1] for x in all_rc])
                box_cnt = Counter([_lt_convert(x) for x in all_rc])
                all_links = []
                for i in range(len(all_rc)):
                    for j in range(i+1, len(all_rc)):
                        ti = all_rc[i]
                        tj = all_rc[j]
                        
                        typ = 'n'
                        if ti[0] == tj[0]:
                            typ = 's' if row_cnt[ti[0]] == 2 else 'w'
                        elif ti[1] == tj[1]:
                            typ = 's' if col_cnt[ti[1]] == 2 else 'w'
                        elif _same_box(ti, tj):
                            typ = 's' if box_cnt[_lt_convert(ti)] == 2 else 'w'
                        if typ != 'n':
                            all_links.append((ti, tj, typ))
                            all_links.append((tj, ti, typ))

                for start_rc in all_rc:
                    chains = []
                    skip_rc = []
                    matched_links = [x for x in all_links if x[0] == start_rc]
                    for link in matched_links:
                        chains.append(tuple([link]))
                        skip_rc.append(tuple([link[1]]))
                    chains = tuple(chains)
                    skip_rc = tuple(skip_rc)

                    while True:
                        new_chains = []
                        new_skip_rc = []
                        found = False
                        old_len = 0
                        for i in range(len(chains)):
                            chain = list(chains[i])
                            curr_skip_rc = list(skip_rc[i])
                            old_len = len(chain)
                            end_rc = chain[-1][1]
                            typ = chain[-1][2]
                            match = [x for x in all_links if x[0] == end_rc and x[2] != typ and x[1] not in curr_skip_rc]
                            for tup in match:
                                new_chains.append(tuple(chain + [tup]))
                                new_skip_rc.append(tuple(curr_skip_rc + [tup[1]]))
                                found = tup[1] == start_rc
                                if found:
                                    loop = new_chains[-1]
                                    if loop[0][2] != loop[-1][2]:
                                        weak_links = [x for x in loop if x[2] == 'w']
                                        loop_rc_lst = new_skip_rc[-1]
                                        for link in weak_links:
                                            elim_tups = []
                                            if link[0][0] == link[1][0]: # row link
                                                row = link[0][0]
                                                for col in self.RCSEQ:
                                                    if (row, col) in loop_rc_lst:
                                                        continue
                                                    if type((v := self.arr[row][col])) == str and char in v:
                                                        elim_tups.append((row, col))
                                            elif link[0][1] == link[1][1]: # col link
                                                col = link[0][1]
                                                for row in self.RCSEQ:
                                                    if (row, col) in loop_rc_lst:
                                                        continue
                                                    if type((v := self.arr[row][col])) == str and char in v:
                                                        elim_tups.append((row, col))
                                            elif _same_box(link[0], link[1]):
                                                ltrow, ltcol = _lt_convert(link[0])
                                                for brow, bcol in self.BSEQ:
                                                    row, col = ltrow + brow, ltcol + bcol
                                                    if (row, col) in loop_rc_lst:
                                                        continue
                                                    if type((v := self.arr[row][col])) == str and char in v:
                                                        elim_tups.append((row, col))
                                            for etup in elim_tups:
                                                elim = self.elim(*etup, char)
                                                elim_cnt += elim[0]
                                                elim_lst += elim[1]
                                    elif loop[0][2] == 's':
                                        elim = self.elim(*loop[0][0], char, invert=True)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                                    elif loop[0][2] == 'w':
                                        elim = self.elim(*loop[0][0], char)
                                        elim_cnt += elim[0]
                                        elim_lst += elim[1]
                                    break
                        chains = tuple(new_chains)
                        skip_rc = tuple(new_skip_rc)
                        if found or len(chains) == 0 or len(chains[0]) == old_len:
                            break
                self.analysis_step_append(elim_lst)
            if self.get_num_psbl_nums() == old:
                break
        if elim_cnt > 0:
            self.analysis_func_append(self.elim_x_cycles)
    
    def solve_add_strategies(self):
        while True:
            old = len(self.added)
            self.solve_singles()
            if not self.remaining:
                break
            if len(self.added) == old:
                self.solve_hidden_singles()
                if not self.remaining:
                    break
                if len(self.added) == old:
                    break
        return self.remaining != 0

    def solve_elim_strategies(self):
        strategies = [
            self.elim_naked_pairs,
            self.elim_naked_triples,
            self.elim_hidden_pairs,
            self.elim_hidden_triples,
            self.elim_naked_quads,
            self.elim_hidden_quads,
            self.elim_intersection_removal,
            self.elim_x_wing,
            self.elim_single_chains,
            self.elim_y_wing,
            self.elim_swordfish,
            self.elim_xyz_wing,
            self.elim_x_cycles,
        ]
        for f in strategies:
            old = self.get_num_psbl_nums()
            f()
            if self.get_num_psbl_nums() != old:
                return True
        return False

    def solve(self):
        while True:
            remaining = self.solve_add_strategies()
            if not remaining:
                break
            try_add = self.solve_elim_strategies()
            if not try_add:
                break
        self.analysis.end_arr = deepcopy(self.arr)
        self.analysis.added = self.added