import numpy as np
import matplotlib
import math
import cv2
import copy

def fade(img, state = 1):
    img1 = (255,255,255) - img 
    img2 = np.uint8(0.5*img1)
    img += img2

def load_PNG(file_name):
    #return 32*32*3 np array
    img = np.array(cv2.imread(file_name, cv2.IMREAD_UNCHANGED))
    mask = img[:,:,3] == 0
    #print(mask)
    img = img[:,:,:3]
    img[mask] = (255,255,255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print("load", img.shape, file_name)
    return img

def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0]//factor, factor, img.shape[1]//factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img

def draw_traj(img, p1, p2, color):
    r = 1
    x0 = int(p1[0])
    y0 = int(p1[1])
    x1 = int(p2[0])
    y1 = int(p2[1])    
    if (x0 == x1 or y1 == y0): ##line is either horizontal or vertical
        xmin = min(x0, x1) - r
        xmax = max(x0, x1) + r
        ymin = min(y0, y1) - r
        ymax = max(y0, y1) + r
        for i in range(int(xmin), int(xmax)):
            for j in range(int(ymin), int(ymax)):
                img[j,i] = color
    else:##line is diagonal
        left = min(p1,p2)
        right = max(p1,p2)
        ydir = 1 if right[1]>left[1] else -1
        
        for i in range(int(left[0]), int(right[0])+1):
            curr_y = int(left[1] + (i-left[0]) * ydir)
            img[curr_y, i] = color
            
    return img

def draw_attention_line(img, p1, p2, color,w):
    # r = 1
    # x0 = int(p1[0])
    # y0 = int(p1[1])
    # x1 = int(p2[0])
    # y1 = int(p2[1])
    # # if (x0 == x1 or y1 == y0): ##line is either horizontal or vertical
    # xmin = min(x0, x1)
    # xmax = max(x0, x1)
    # ymin = min(y0, y1)
    # ymax = max(y0, y1)
    # for i in range(int(xmin), int(xmax)):
    #     for j in range(int(ymin), int(ymax)):
    #         img[j,i] = color
    left = min(p1, p2)
    right = max(p1, p2)
    # ydir = 1 if right[1] > left[1] else -1
    # if
    theta =math.atan((right[1] -left[1])/(right[0] -left[0]+0.0001))
    ydir = math.tan(theta)

    for i in range(int(left[0]), int(right[0]) + 1):
        curr_y = int(left[1] + (i - left[0]) * ydir)
        img[curr_y, i] = color
        if w==2:
            img[curr_y+1, i+1] = color
            img[curr_y-1, i-1] = color
        elif w==1:
            img[curr_y + 1, i + 1] = color
    return img

def fill_coords(img, fn, color, state = -1):
    """
    Fill pixels of an image with coordinates matching a filter function
    """
    #if (state >= 3): return img
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                if state == 1:
                    img[y, x] = (127,127,127) + color/2
                elif state == 2:
                    img[y, x] = (191,191,191) + color/4
                elif state >= 3:
                    img[y, x] = (191,191,191) + color/4
                else:
                    img[y,x] = color
    return img

def point_on_side(width):
    def fn(x, y):
        return x<width or x>1-width or y<width or y>1-width
    return fn

def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout

def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn

def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
    return fn

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

def highlight_img(img, alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img - alpha * img
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
    return img
