import math

from rich.text import Text


class BrailleCanvas:
    """
    Zeichnet auf einem Braille-Raster.
    Jede Terminal-Zelle = 2x4 Braille-Dots -> 2x horizontale, 4x vertikale Aufloesung.
    """

    BRAILLE_BASE = 0x2800
    DOT_MAP = {
        (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20,
        (0, 3): 0x40, (1, 3): 0x80,
    }

    def __init__(self, char_width: int, char_height: int):
        self.char_width = char_width
        self.char_height = char_height
        self.px_width = char_width * 2
        self.px_height = char_height * 4
        self._buf = [[0] * char_width for _ in range(char_height)]
        self._color_buf = [[None] * char_width for _ in range(char_height)]

    def clear(self):
        for row in self._buf:
            for i in range(len(row)):
                row[i] = 0
        for row in self._color_buf:
            for i in range(len(row)):
                row[i] = None

    def set_pixel(self, px: int, py: int, color: str = None):
        if px < 0 or px >= self.px_width or py < 0 or py >= self.px_height:
            return
        char_col = px // 2
        char_row = py // 4
        sub_col = px % 2
        sub_row = py % 4
        bit = self.DOT_MAP.get((sub_col, sub_row), 0)
        self._buf[char_row][char_col] |= bit
        if color:
            self._color_buf[char_row][char_col] = color

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: str = None):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_thick_line(self, x0: int, y0: int, x1: int, y1: int,
                        thickness: int = 2, color: str = None):
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            self.set_pixel(x0, y0, color)
            return

        nx = -dy / length
        ny = dx / length

        for t in range(-(thickness // 2), thickness // 2 + 1):
            ox = int(nx * t)
            oy = int(ny * t)
            self.draw_line(x0 + ox, y0 + oy, x1 + ox, y1 + oy, color)

    def draw_circle(self, cx: int, cy: int, r: int, color: str = None):
        x = r
        y = 0
        err = 1 - r

        while x >= y:
            for px, py in [(cx+x, cy+y), (cx-x, cy+y), (cx+x, cy-y), (cx-x, cy-y),
                           (cx+y, cy+x), (cx-y, cy+x), (cx+y, cy-x), (cx-y, cy-x)]:
                self.set_pixel(px, py, color)
            y += 1
            if err < 0:
                err += 2 * y + 1
            else:
                x -= 1
                err += 2 * (y - x) + 1

    def fill_circle(self, cx: int, cy: int, r: int, color: str = None):
        for dy in range(-r, r + 1):
            dx = int(math.sqrt(r * r - dy * dy))
            for x in range(cx - dx, cx + dx + 1):
                self.set_pixel(x, cy + dy, color)

    def draw_ellipse_arc(self, cx: int, cy: int, rx: int, ry: int,
                         start_angle: float, end_angle: float,
                         steps: int = 60, color: str = None):
        for i in range(steps):
            t = start_angle + (end_angle - start_angle) * i / steps
            x = int(cx + rx * math.cos(t))
            y = int(cy + ry * math.sin(t))
            self.set_pixel(x, y, color)

    def render(self) -> list[Text]:
        lines = []
        for row_idx in range(self.char_height):
            text = Text()
            for col_idx in range(self.char_width):
                bits = self._buf[row_idx][col_idx]
                char = chr(self.BRAILLE_BASE + bits) if bits else ' '
                color = self._color_buf[row_idx][col_idx]
                if color and bits:
                    text.append(char, style=color)
                else:
                    text.append(char)
            lines.append(text)
        return lines
