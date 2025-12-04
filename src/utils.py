import math

def convert_quad_to_box(quad):
    if not quad or len(quad) < 8:
        return [0, 0, 0, 0]
    xs = quad[0::2]
    ys = quad[1::2]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]

def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    width = max(1, int(width))
    height = max(1, int(height))

    x0 = min(max(0, x0), width)
    x1 = min(max(0, x1), width)
    y0 = min(max(0, y0), height)
    y1 = min(max(0, y1), height)

    def norm_x(x): return int(round((x / width) * 1000))
    def norm_y(y): return int(round((y / height) * 1000))

    nx0, ny0, nx1, ny1 = norm_x(x0), norm_y(y0), norm_x(x1), norm_y(y1)

    nx0 = min(max(0, nx0), 1000)
    ny0 = min(max(0, ny0), 1000)
    nx1 = min(max(0, nx1), 1000)
    ny1 = min(max(0, ny1), 1000)

    if nx1 < nx0: nx0, nx1 = nx1, nx0
    if ny1 < ny0: ny0, ny1 = ny1, ny0
    return [nx0, ny0, nx1, ny1]