from itertools import product

from bokeh.plotting import figure, show, curdoc
from bokeh.io import output_notebook
from bokeh.layouts import grid, column
from bokeh.models import SingleIntervalTicker, BoxAnnotation, ColumnDataSource, Select

output_notebook()

import cairo

class Analysis:
    def __init__(self, init_num_psbl, init_arr=None, end_arr=None, added=None):
        self.seq = []
        self.init_num_psbl = init_num_psbl

        self.init_arr = init_arr
        self.init_arr_name = None
        self.end_arr = end_arr
        self.end_arr_name = None

        self.added = added
        self.steps = [[]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        o = ''
        for t in self.seq:
            o += f'Func: {" ".join([s.capitalize() for s in (t[0].__name__).split("_")[1:]])} - {t[1]} nums - {t[2]:.2g} ({t[2]}) nodes\n'
        return o[:-1]

    def func_append(self, func, nums, nodes):
        self.seq.append((func, nums, nodes))

    def board_str(self, arr):
        o = ''
        for row, col in product(range(9), repeat=2):
            o += '0' if type((v := arr[row][col])) == str else str(v)
        return o

    def clean_steps_arr(self):
        for step in range(len(self.steps)):
            for substep in range(len(self.steps[step])):
                self.steps[step][substep] = [x for x in self.steps[step][substep] if x is not None]
        for step in range(len(self.steps)):
            self.steps[step] = [x for x in self.steps[step] if len(x) > 0]

    def base_1000_endcode(self, s):
        code = ''.join([chr(x) for x in range(48, 58)] + [chr(x) for x in range(97, 123)])
        encoded = ''
        for i in range(0, len(s), 3):
            sv = int(s[i:i+3])
            first = sv % len(code)
            second = sv // len(code)
            encoded += f'{code[second]}{code[first]}'
        return encoded

    def generate_image(self, ie):
        path = './images/'
        arr_name = f'{self.base_1000_endcode(self.board_str(self.init_arr))}.png'
        if ie == 'i':
            arr = self.init_arr
            arr_name = path + arr_name
            self.init_arr_name = arr_name
        elif ie == 'e':
            arr = self.end_arr
            arr_name = path + 's_' + arr_name
            self.end_arr_name = arr_name

        window_size = 720
        width, height = window_size, window_size

        # rgba values
        WHITE = (1, 1, 1, 1)
        BLACK = (0, 0, 0, 1)
        BLUE = (0, 0, 1, 1)
        GRAY = (0, 0, 0, .5)

        bg_color = WHITE
        border_color = BLACK

        given_color = BLACK
        solved_color = BLUE
        psbl_color = GRAY
        
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        ctx.scale(width, height)
        
        width, height = 1, 1

        border_thickness = .025
        thick_thickness = 2 / 5 * border_thickness
        thin_thickness = thick_thickness / 2

        ctx.rectangle(0, 0, 1, 1)
        ctx.set_source_rgba(*bg_color)
        ctx.fill_preserve()
        ctx.set_source_rgba(*border_color)
        ctx.set_line_width(border_thickness)
        ctx.stroke()

        eff_width, eff_height = width - border_thickness, height - border_thickness

        for i in range(1, 9):
            if i in [3, 6]:
                ctx.set_line_width(thick_thickness)
            else:
                ctx.set_line_width(thin_thickness)

            ctx.move_to(border_thickness / 2 + i * eff_width / 9, 0)
            ctx.rel_line_to(0, height)
            ctx.stroke()

            ctx.move_to(0, border_thickness / 2 + i * eff_height / 9)
            ctx.rel_line_to(width, 0)
            ctx.stroke()

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        given_glyphs = []
        solved_glyphs = []
        psbl_glyphs = []

        all_lines_thickness = 6 * thin_thickness + 2 * thick_thickness
        sbox_width = (eff_width - all_lines_thickness) / 9
        sbox_height = (eff_height - all_lines_thickness) / 9

        wnd_x_start = wnd_y_start = border_thickness / 2

        large_font_size = .08
        small_font_size = .03

        for row, col in product(range(9), repeat=2):
            box_x_start = wnd_x_start + eff_width * col / 9
            box_y_start = wnd_y_start + eff_height * (row + .5) / 9
            box_x_start += thick_thickness / 2 if col in [3, 6] else thin_thickness / 2
            box_y_start += thick_thickness / 2 if row in [3, 6] else thin_thickness / 2

            box_x_ofs = eff_width / 36
            box_y_ofs = eff_height / 36

            sbox_x_start_ofs = sbox_width / 9
            sbox_y_start_ofs = sbox_height / 9 

            eff_sbox_width = sbox_width - 2 * sbox_x_start_ofs
            eff_sbox_height = sbox_height - 2 * sbox_y_start_ofs

            if type((v := arr[row][col])) == str:
                if v == '':
                    raise ValueError('Invalid Board')
                for char in v:
                    val = int(char)
                    srow, scol = (val - 1) // 3, (val - 1) % 3

                    sbox_x_start = box_x_start + sbox_x_start_ofs + eff_sbox_width * scol / 3
                    sbox_y_start = box_y_start + sbox_x_start_ofs + eff_sbox_height * srow / 3

                    sbox_x_ofs = sbox_width / 24
                    sbox_y_ofs = -sbox_height / 6 - box_y_ofs / 2

                    psbl_glyphs.append((19 + val, sbox_x_start + sbox_x_ofs, sbox_y_start + sbox_y_ofs))
            else:                
                t = (19 + v, box_x_start + box_x_ofs, box_y_start + box_y_ofs)
                if (row, col) in self.added:
                    solved_glyphs.append(t)
                else:
                    given_glyphs.append(t)

        ctx.set_font_size(large_font_size)

        ctx.glyph_path(given_glyphs)
        ctx.set_source_rgba(*given_color)
        ctx.fill()

        ctx.glyph_path(solved_glyphs)
        ctx.set_source_rgba(*solved_color)
        ctx.fill()

        ctx.set_font_size(small_font_size)

        ctx.glyph_path(psbl_glyphs)
        ctx.set_source_rgba(*psbl_color)
        ctx.fill()

        surface.write_to_png(arr_name)

    def plot(self):
        self.generate_image('i')
        self.generate_image('e')

        width = 650

        b = figure(
            width=width,
            height=width//2-width//15,
            match_aspect=True,
            tools='',
        )
        b.axis.visible = False
        b.grid.grid_line_color = None
        b.outline_line_color = None
        b.toolbar.logo = None
        b.toolbar_location = None

        b.image_url(url=[self.init_arr_name], x=0, y=1, w=1.25, h=1.25)
        b.image_url(url=[self.end_arr_name], x=1.4, y=1, w=1.25, h=1.25)

        num_psbl = [self.init_num_psbl] + [t[1] for t in self.seq]
        steps = list(range(len(num_psbl)))
        names = ['Start'] + [' '.join([s.capitalize() for s in (t[0].__name__).split('_')[1:]]) for t in self.seq]

        if 'Xyz Wing' in names:
            names[names.index('Xyz Wing')] = 'XYZ Wing'

        if len(self.seq) > 0:
            elim_cnt = [0, self.init_num_psbl - self.seq[0][1]]
        else:
            elim_cnt = [0]
        for i in range(1, len(self.seq)):
            elim_cnt.append(self.seq[i-1][1] - self.seq[i][1])

        data = {
            'x': steps,
            'y': num_psbl,
            'elim_cnt': elim_cnt,
            'names': names,
        }
        g_source = ColumnDataSource(data=data)

        g_tooltips = [('Func', '@names'), ('# elim', '@elim_cnt'), ('# psbl', '@y')]

        g = figure(
            title='Sudoku Solver Steps',
            x_axis_label='Steps',
            y_axis_label='Number of possibilities',
            tooltips=g_tooltips,
            width=width,
            height=250,
        )
        g.title.align = 'center'

        g.xaxis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=0)
        g.axis.axis_label_text_font_style = 'normal'

        g.line(x='x', y='y', line_width=2, source=g_source)
        g.circle(x='x', y='y', source=g_source)

        func_type = [t[0].__name__.split('_')[0] for t in self.seq]
        for i in range(len(steps)-1):
            if func_type[i] == 'solve':
                g.add_layout(BoxAnnotation(left=i, right=i+1, fill_alpha=.2, fill_color='green'))
            elif func_type[i] == 'elim':
                g.add_layout(BoxAnnotation(left=i, right=i+1, fill_alpha=.2, fill_color='yellow'))

        self.clean_steps_arr()

        step_graphs = []

        for step in range(len(self.steps)-1):
            a_steps, a_flatidx, a_val = [], [], []
            e_steps, e_flatidx, e_val = [], [], []
            ae_steps, ae_flatidx, ae_val = [], [], []
            ce_steps, ce_flatidx, ce_val = [], [], []
            ca_steps, ca_flatidx, ca_val = [], [], []
            cae_steps, cae_flatidx, cae_val = [], [], []

            for substep in range(len(self.steps[step])):
                for t in self.steps[step][substep]:
                    flatidx = t[1] * 9 + t[2]
                    if t[0] == 'a':
                        a_steps.append(substep+1)
                        a_flatidx.append(flatidx)
                        a_val.append(t[3])
                    elif t[0] == 'e':
                        e_steps.append(substep+1)
                        e_flatidx.append(flatidx)
                        e_val.append(t[3])
                    elif t[0] == 'ae':
                        ae_steps.append(substep+1)
                        ae_flatidx.append(flatidx)
                        ae_val.append(t[3])
                    elif t[0] == 'ce':
                        ce_steps.append(substep+1)
                        ce_flatidx.append(flatidx)
                        ce_val.append(t[3])
                    elif t[0] == 'ca':
                        ca_steps.append(substep+1)
                        ca_flatidx.append(flatidx)
                        ca_val.append(t[3])
                    elif t[0] == 'cae':
                        cae_steps.append(substep+1)
                        cae_flatidx.append(flatidx)
                        cae_val.append(t[3])

            a_idx = [(x//9+1, x%9+1) for x in a_flatidx]
            e_idx = [(x//9+1, x%9+1) for x in e_flatidx]
            ae_idx = [(x//9+1, x%9+1) for x in ae_flatidx]
            ce_idx = [(x//9+1, x%9+1) for x in ce_flatidx]
            ca_idx = [(x//9+1, x%9+1) for x in ca_flatidx]
            cae_idx = [(x//9+1, x%9+1) for x in cae_flatidx]

            a_data = {
                'x': a_steps,
                'y': a_flatidx,
                'idx': a_idx,
                'a_val': a_val,
                'e_val': ['-'] * len(a_steps),
                
            }
            a_source = ColumnDataSource(data=a_data)

            e_data = {
                'x': e_steps,
                'y': e_flatidx,
                'idx': e_idx,
                'a_val': ['-'] * len(e_steps),
                'e_val': e_val,
            }
            elim_source = ColumnDataSource(data=e_data)

            ae_data = {
                'x': ae_steps,
                'y': ae_flatidx,
                'idx': ae_idx,
                'a_val': [x['a'] for x in ae_val],
                'e_val': [x['e'] for x in ae_val],
            }
            ae_source = ColumnDataSource(data=ae_data)

            ce_data = {
                'x': ce_steps,
                'y': ce_flatidx,
                'idx': ce_idx,
                'a_val': ['-'] * len(ce_steps),
                'e_val': ce_val,
            }
            ce_source = ColumnDataSource(data=ce_data)

            ca_data = {
                'x': ca_steps,
                'y': ca_flatidx,
                'idx': ca_idx,
                'a_val': ca_val,
                'e_val': ['-'] * len(ca_steps),
                
            }
            ca_source = ColumnDataSource(data=ca_data)

            cae_data = {
                'x': cae_steps,
                'y': cae_flatidx,
                'idx': cae_idx,
                'a_val': [x['a'] for x in cae_val],
                'e_val': [x['e'] for x in cae_val],
                
            }
            cae_source = ColumnDataSource(data=cae_data)

            s_tooltips = [('Substep', '@x'), ('Idx', '(@idx)'), ('Add', '[@a_val]'), ('Elim', '[@e_val]')]

            s = figure(
                title=f'Step {step+1} - {names[step+1]} - {elim_cnt[step+1]} elim',
                x_axis_label='Sub-Steps',
                y_axis_label='Flat Index',
                tooltips=s_tooltips,
                width=width,
                height=250
            )
            s.title.align = 'center'

            s.xaxis.ticker.min_interval = 1.0
            s.axis.axis_label_text_font_style = 'normal'

            x_max = 0

            if len(a_steps) > 0:
                s.circle(x='x', y='y', source=a_source, color='green', legend_label='A')
                x_max = max(x_max, a_steps[-1])
            if len(e_steps) > 0:
                s.circle(x='x', y='y', source=elim_source, color='red', legend_label='E')
                x_max = max(x_max, e_steps[-1])
            if len(ae_steps) > 0:
                s.circle(x='x', y='y', source=ae_source, color='blue', legend_label='AE')
                x_max = max(x_max, ae_steps[-1])
            if len(ce_steps) > 0:
                s.circle(x='x', y='y', source=ce_source, color='orange', legend_label='CE')
                x_max = max(x_max, ce_steps[-1])
            if len(ca_steps) > 0:
                s.circle(x='x', y='y', source=ca_source, color='magenta', legend_label='CA')
                x_max = max(x_max, ca_steps[-1])
            if len(cae_steps) > 0:
                s.circle(x='x', y='y', source=cae_source, color='brown', legend_label='CAE')
                x_max = max(x_max, cae_steps[-1])

            x_max += x_max // 2.7
            if x_max < 7:
                x_max += 1

            s.x_range.end = x_max
            s.y_range.end = 85

            s.legend.location = 'top_right'
            s.legend.orientation = 'vertical'
            s.legend.background_fill_alpha = 0
            s.legend.click_policy = 'mute'
            s.legend.label_text_font_size = '10px'
            s.legend.glyph_height = 10
            s.legend.label_height = 5
            s.legend.label_width = 5
            s.legend.label_standoff = 0

            if max(max(a_steps, default=0), max(e_steps, default=0), max(ae_steps, default=0)) > 6:
                step_graphs.append(s)
            else:
                if len(step_graphs) == 0:
                    step_graphs.append([s])
                elif type(step_graphs[-1]) == list and len(step_graphs[-1]) < 2:
                    step_graphs[-1].append(s)
                else:
                    step_graphs.append([s])

        step_graphs = [x if type(x) == list else [x] for x in step_graphs]
        for l in step_graphs:
            for s in l:
                s.width //= len(l)

        step_titles = []
        for l in step_graphs:
            for s in l:
                step_titles.append(s.title.text)
        if len(step_titles) == 0:
            step_titles.append('')

        p = grid([
            [b],
            [g],
            *step_graphs,
        ])
        show(p)