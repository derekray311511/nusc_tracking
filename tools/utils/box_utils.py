import random
import numpy as np
import math
from matplotlib import pyplot as plt

def box3d_filter(box3d, points):
    '''
    Input:
        `box3d` = [bx1, by1, bz1, bx2, by2, bz2]\n
        `points` = list of 3d points\n
    Return:
        `points` in the box\n
        `points` out of the box\n
        `idx` in the box\n
    '''
    bx1, bx2 = sorted([box3d[0], box3d[3]])
    by1, by2 = sorted([box3d[1], box3d[4]])
    bz1, bz2 = sorted([box3d[2], box3d[5]])

    pts = np.array(points)
    ll = np.array([bx1, by1, bz1])  # lower-left
    ur = np.array([bx2, by2, bz2])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    outbox = pts[np.logical_not(inidx)]

    idx = np.where(inidx)[0]

    return inbox, outbox, idx

def box2d_filter(box2d, points):
    '''
    Input:
        `box2d` = [bx1, by1, bx2, by2]\n
        `points` = list of 2d points\n
    Return:
        `points` in the box\n
        `points` out of the box\n
        `idx` in the box\n
    '''
    bx1, bx2 = sorted([box2d[0], box2d[2]])
    by1, by2 = sorted([box2d[1], box2d[3]])

    pts = np.array(points)
    ll = np.array([bx1, by1])  # lower-left
    ur = np.array([bx2, by2])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    outbox = pts[np.logical_not(inidx)]

    idx = np.where(inidx)[0]

    return inbox, outbox, idx

def is_points_inside_obb(points, center, size, angle_rad):
    """ 
    Returns a boolean array indicating whether each point is inside the OBB

    Params:
        points: list of [x, y]
        center: box center [x, y]
        size  : size of box [vx, vy]
        angle_rad : heading of the box (in radians)
    """
    points = np.array(points)
    center = np.array(center)
    width, height = size[0], size[1]

    # Compute the transformation matrix to align the OBB with the coordinate axes
    transform_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])

    # Transform the points to the OBB coordinate system
    transformed_points = points - center
    transformed_points = np.dot(transformed_points, transform_matrix)

    # Calculate the half-width and half-height of the OBB
    half_width = width / 2
    half_height = height / 2

    # Check if each point is inside the OBB
    is_inside = (np.abs(transformed_points[:, 0]) <= half_width) & (np.abs(transformed_points[:, 1]) <= half_height)

    return is_inside

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                    [s,  c,  0],
                    [0, 0,  1]])

def get_3d_box_2corner(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:heading_angle
        box_size: tuple of (l,w,h)
        : rad scalar, clockwise from pos z axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (2,3) for 3D box cornders
    '''
    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l/2, -l/2]
    y_corners = [w/2, -w/2]
    z_corners = [h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    bx1, bx2 = sorted([corners_3d[0][0], corners_3d[1][0]])
    by1, by2 = sorted([corners_3d[0][1], corners_3d[1][1]])
    bz1, bz2 = sorted([corners_3d[0][2], corners_3d[1][2]])
    corners_3d = np.array([[bx1, by1, bz1],
                          [bx2, by2, bz2]])
    return corners_3d

def get_3d_box_8corner(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:heading_angle
        box_size: tuple of (l,w,h)
        : rad scalar, clockwise from pos z axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def test2d():
    points = [(random.random(), random.random()) for i in range(100)]

    bx1, bx2 = sorted([random.random(), random.random()])
    by1, by2 = sorted([random.random(), random.random()])
    box = [bx1, by1, bx2, by2]

    inbox, outbox, _ = box2d_filter(box, points)

    # this is just for drawing
    rect = np.array([[bx1, by1], [bx1, by2], [bx2, by2], [bx2, by1], [bx1, by1]])

    plt.plot(inbox[:, 0], inbox[:, 1], 'rx',
                outbox[:, 0], outbox[:, 1], 'bo',
                rect[:, 0], rect[:, 1], 'g-')
    plt.show()

def test3d():
    points = [(random.random(), random.random(), random.random()) for i in range(100)]

    bx1, bx2 = sorted([random.random(), random.random()])
    by1, by2 = sorted([random.random(), random.random()])
    bz1, bz2 = sorted([random.random(), random.random()])
    box = [bx1, by1, bz1, bx2, by2, bz2]

    inbox, outbox, _ = box3d_filter(box, points)

    # this is just for drawing
    rect = np.array([[bx1, by1], [bx1, by2], [bx2, by2], [bx2, by1], [bx1, by1]])

    plt.plot(inbox[:, 0], inbox[:, 1], 'rx',
                outbox[:, 0], outbox[:, 1], 'bo',
                rect[:, 0], rect[:, 1], 'g-')
    plt.show()

import numpy as np
from scipy.spatial import ConvexHull

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0
    
def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                    [s,  c,  0],
                    [0, 0,  1]])

def get_3d_box(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:heading_angle
        box_size: tuple of (l,w,h)
        : rad scalar, clockwise from pos z axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box corners
    '''
    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_2d_box(center, box_size, heading_angle):
    ''' Calculate 2D bounding box corners from its parameterization.

    Input:heading_angle
        box_size: tuple of (l,w)
        : rad scalar, clockwise from pos z axis
        center: tuple of (x,y)
    Output:
        corners_2d: numpy array of shape (4,2) for 2D box corners
    '''
    R = rotz(heading_angle)[:2, :2]
    l, w = box_size
    x_corners = [l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2]
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + center[0]
    corners_2d[1, :] = corners_2d[1, :] + center[1]
    corners_2d = np.transpose(corners_2d)
    return corners_2d

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0,2], corners2[0,2])
    ymin = max(corners1[4,2], corners2[4,2])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def q_to_xyzw(Q):
    '''
    wxyz -> xyzw
    '''
    return [Q[1], Q[2], Q[3], Q[0]]

def euler_from_quaternion(q):
    """
    x y z w -> roll pitch yaw \n
    Convert a quaternion into euler angles (roll, pitch, yaw) \n
    roll is rotation around x in radians (counterclockwise) \n
    pitch is rotation around y in radians (counterclockwise) \n
    yaw is rotation around z in radians (counterclockwise) \n
    """
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def nms(boxes, iou_th):
    '''
    each box: [x, y, z, dx, dy, dz, yaw, vx, vy, name, score]
    '''    
    # Transform to 8 corners
    corners = []
    scores = []
    for box in boxes:
        x, y, z = box['translation']
        dx, dy, dz = box['size']    # l w h
        Quaternion = q_to_xyzw(box['rotation'])
        roll, pitch, yaw = euler_from_quaternion(Quaternion)
        corner = get_3d_box([x, y, z], [dy, dx, dz], yaw)
        corners.append(corner)
        scores.append(box['detection_score'])
    corners = np.array(corners)
    scores = np.array(scores)

    # Do nms
    boxes.sort(reverse=True, key=lambda box:box['detection_score'])
    corners_sorted = corners[np.flip(np.argsort(scores))]
    # scores = scores[np.flip(np.argsort(scores))]
    box_indices = np.arange(0, len(corners_sorted))
    suppressed_box_indices = []
    tmp_suppress = []

    while len(box_indices) > 0:

        if box_indices[0] not in suppressed_box_indices:
            selected_box = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_box:
                    iou3d, selected_iou = box3d_iou(corners_sorted[selected_box], corners_sorted[box_indices[i]])
                    if selected_iou > iou_th:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        box_indices = box_indices[1:]

    # corners_sorted = np.delete(corners_sorted, suppressed_box_indices, axis=0)
    preserved_boxes = boxes
    preserved_boxes = [preserved_boxes[i] for i in range(len(preserved_boxes)) if i not in suppressed_box_indices]
    return preserved_boxes, suppressed_box_indices

def nms_visualizer(nms, *args):
    dets, iou = args[0], args[1]

    def box2corners2d(boxes):
        corners = []
        for box in dets:
            x, y, z = box['translation']
            dx, dy, dz = box['size']
            Quaternion = q_to_xyzw(box['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            corner = get_3d_box([x, y, z], [dy, dx, dz], yaw)
            corners.append(corner[:4, :2])
        return np.array(corners)
    
    def draw_boxes(ax, corners, color, thickness=2):
        corners = corners[:, [0, 1, 2, 3, 0], :2]
        print(f"corners shape: {corners.shape}")
        for index in range(corners.shape[0]):
            ax.plot(
                corners[index, :, 0],
                corners[index, :, 1],
                linewidth=thickness,
                color=np.array(color) / 255,
            )
    

    corners2d = box2corners2d(dets)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    draw_boxes(ax, corners2d, color=[66, 135, 245], thickness=4)
    dets, suppressed_box_indices = nms(dets, iou)
    
    corners2d = box2corners2d(dets)
    draw_boxes(ax, corners2d, color=[245, 179, 66], thickness=1.5)
    plt.tight_layout()
    plt.show()

    return dets, suppressed_box_indices


if __name__ == "__main__":
    points = np.array([[2, 1], [0, 0], [3, 4]])  # Example points
    center = np.array([1, 2])  # OBB center coordinates
    size = (4, 2) # OBB width, height
    angle = 0  # OBB rotation angle in degrees

    inside_mask = is_points_inside_obb(points, center, size, angle)
    print(inside_mask)