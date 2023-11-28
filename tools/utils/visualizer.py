import cv2
import numpy as np
import itertools
import colorsys
import time

from .geometry_utils import *
from .box_utils import get_3d_box, get_2d_box
from .utils import encodeCategory, decodeCategory, get_trk_colormap
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

class TrackVisualizer:
    def __init__(
        self, 
        viz_cat: list,
        windowName: str = "track",
        range_: tuple = (100, 100),
        windowSize: tuple = (800, 800),
        imgSize: tuple = (1600, 1600), 
        duration: float = 0.5,
        background_color: tuple = (50, 50, 50),
        grid: bool = True,
        **kwargs,
    ):
        self.viz_cat = viz_cat
        self.trk_colorMap = get_trk_colormap()
        self.range = range_
        self.height = imgSize[0]
        self.width = imgSize[1]
        self.resolution = self.range[0] / self.height
        self.play = True
        self.duration = duration
        self.windowName = windowName
        self.window = cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.grid = grid
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.background_color
        
        cv2.resizeWindow(self.windowName, windowSize)
        print(f"window: {self.windowName}")
        print(f"Visualize category: {self.viz_cat}")
        print(f"Visualize range: {self.range}")
        print(f"res: {self.resolution}")
        print(f"Image size: {self.height, self.width}")
        print(f"duration: {self.duration}")

    def reset(self):
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.background_color

    def draw_ego_car(self, img_src):
        # Load the car image with an alpha channel (transparency)
        car_image = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)
        alpha_channel = car_image[:, :, 3] / 255.0
        car_image = car_image[:, :, :3]
        car_image = car_image * alpha_channel[:, :, np.newaxis]
        img = car_image.astype(np.uint8)

        carSize = 6 # meters
        H, W = img.shape[:2]
        new_H = int(carSize / self.resolution) + int(carSize / self.resolution) % 2
        new_W = int(new_H * H / W) + int(new_H * H / W) % 2
        img = cv2.resize(img, (new_H, new_W))
        y, x = self.height // 2, self.width // 2
        roi = self.image[y - img.shape[0] // 2 : y + img.shape[0] // 2, x - img.shape[1] // 2 : x + img.shape[1] // 2]
        result = cv2.add(roi, img)
        self.image[y - img.shape[0] // 2:y + img.shape[0] // 2, x - img.shape[1] // 2:x + img.shape[1] // 2] = result

    def draw_points(self, pts: np.ndarray, BGRcolor=(50, 50, 255)):
        for p in pts:
            p = np.round(p).astype(int)
            cv2.circle(self.image, p, 4, BGRcolor, -1)

    def draw_bboxes(self, corners: np.ndarray, BGRcolor=(255, 150, 150), thickness=2, alpha=1.0, **kwargs):
        if alpha == 1.0:
            for box in corners:
                box = np.round(box).astype(int)
                cv2.drawContours(self.image, [box], 0, BGRcolor, thickness=thickness)
        else:
            boxes_image = np.zeros_like(self.image)
            for box in corners:
                box = np.round(box).astype(int)
                cv2.drawContours(boxes_image, [box], 0, BGRcolor, thickness=thickness)
            self.image = cv2.addWeighted(self.image, 1, boxes_image, alpha, 0)

    def draw_radar_pts(self, radar_pc: list, trans: np.ndarray, BGRcolor=(50, 50, 255), showContours=False):
        """Param :

        radar_pc : list of [x, y, z, vx, vy]
        ego_pose : {'translation': [x, y, z], 'rotation': [w, x, y, z], 'timestamp': t, 'token' : t}

        """
        if len(radar_pc) == 0:
            return

        local_pts = []
        for point in radar_pc:
            local_pts.append(point[:3])

        local_pts = np.array(local_pts, dtype=float)
        local_pts = np.hstack([local_pts, np.ones((local_pts.shape[0], 1), dtype=float)])
        local_pts = (trans @ local_pts.T).T[:, :2] / self.resolution
        local_pts = local_pts * np.array([1, -1])
        local_pts = local_pts + np.array([self.height // 2, self.width // 2])
        self.draw_points(local_pts, BGRcolor)
        if showContours:
            convex_contour = cv2.convexHull(np.array(local_pts, dtype=int))
            cv2.drawContours(self.image, [convex_contour], 0, BGRcolor, 2)

    def draw_radar_seg(
        self, 
        radarSeg: np.ndarray, 
        trans: np.ndarray, 
        colorID=False, 
        colorName=False, 
        contours=True, 
        **kwargs
    ):
        if colorID and colorName:
            assert "colorID and colorName can not be True simultaneously"
        if colorID:
            for k, g in itertools.groupby(radarSeg, lambda x: x[5]):
                g = list(g)
                # BGRcolor = getColorFromID(ID=k, colorRange=(50, 255))
                BGRcolor = getColorFromID_HSV(ID=k, cycle_num=12)
                if k == -1:  # id == -1
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, showContours=False)
                else:
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, showContours=contours)
        elif colorName:
            for k, g in itertools.groupby(radarSeg, lambda x: x[5]):
                g = list(g)
                cat_num = int(g[0][6])
                cat_name = decodeCategory([cat_num], self.viz_cat)[0]
                if cat_num == -1:  # id == -1
                    B, G, R = 100, 100, 100 # Gray color
                    self.draw_radar_pts(g, trans, BGRcolor=(B, G, R), showContours=False)
                else:
                    BGRcolor = self.trk_colorMap[cat_name]
                    self.draw_radar_pts(g, trans, BGRcolor=BGRcolor, showContours=contours)
        else:
            self.draw_radar_pts(radarSeg, trans) 

    def world2ego(self, objects, ego_trans):
        ret = []
        for object in objects:
            trans = np.array(object['translation'])
            vel = np.array([object['velocity'][0], object['velocity'][1], 0.0])
            rot = quaternion_rotation_matrix(object['rotation'])
            trans = np.hstack([rot, trans.reshape(-1, 1)])
            trans = np.vstack([trans, np.array([0, 0, 0, 1])]).reshape(-1, 4)
            vel = vel.reshape(-1, 1)
            new_trans = ego_trans @ trans
            new_vel = ego_trans[:3, :3] @ vel
            object['translation'] = new_trans[:3, 3].ravel().tolist()
            object['rotation'] = q_to_wxyz(R.from_matrix(new_trans[:3, :3]).as_quat())
            object['velocity'] = new_vel.ravel()[:2]
            ret.append(object)
        return ret

    def getBoxCorners2d(self, boxes: list) -> np.ndarray:
        corners = []
        for box in boxes:
            x, y, z = box['translation']
            w, l, h = box['size']
            Quaternion = q_to_xyzw(box['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            corner = get_2d_box([x, y], [l, w], yaw)
            corners.append(corner)
        return np.array(corners)

    def draw_det_bboxes(
        self, 
        nusc_det: list, 
        trans: np.ndarray, 
        draw_vel: bool = False,
        draw_id: bool = False,
        BGRcolor=(255, 150, 150), 
        thickness=2,
        colorID=False, 
        colorName=False, 
        **kwargs
    ):
        """Param :

        nusc_det : list of {'translation': [x, y, z], 'rotation': [w, x, y, z], 'size': [x, y, z], 'velocity': [vx, vy], 'detection_name': s, 'detection_score': s, 'sample_token': t}
        ego_pose : {'translation': [x, y, z], 'rotation': [w, x, y, z], 'timestamp': t, 'token' : t}

        """
        if len(nusc_det) == 0:
            return
        nusc_det = deepcopy(nusc_det)
        nusc_det = self.world2ego(nusc_det, trans)
        for det in nusc_det:
            # Nusc to image x, y coordinates (Flip y axis)
            det['translation'][:2] *= np.array([1, -1])
            det['size'][:2] *= np.array([1, -1])
            det['velocity'][:2] *= np.array([1, -1])
            det['rotation'][0] *= -1
            # Transform to image pixel level
            det['translation'] = np.array(det['translation']) / self.resolution
            det['translation'][:2] = det['translation'][:2] + np.array([self.height // 2, self.width // 2])
            det['size'] = np.array(det['size']) / self.resolution
            det['velocity'] = np.array(det['velocity']) / self.resolution

        if colorName:   # Draw boxes by detection_name
            for k, g in itertools.groupby(nusc_det, lambda x: x['detection_name']):
                g_det = list(g)
                cat_num = encodeCategory([k], self.viz_cat)[0]
                BGRcolor = self.trk_colorMap[k]
                corners2d = self.getBoxCorners2d(g_det)
                self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
                if draw_vel:
                    self._draw_vel(g_det, BGRcolor, thickness, **kwargs)
                if draw_id:
                    self._draw_id(g_det, BGRcolor, **kwargs)
        elif colorID and ('tracking_id' in nusc_det[0]):
            for det in nusc_det:
                # BGRcolor = getColorFromID(ID=det['tracking_id'], colorRange=(50, 255))
                BGRcolor = getColorFromID_HSV(ID=det['tracking_id'], cycle_num=12)
                corners2d = self.getBoxCorners2d([det])
                self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
                if draw_vel:
                    self._draw_vel([det], BGRcolor, thickness, **kwargs)
                if draw_id:
                    self._draw_id([det], BGRcolor, **kwargs)
        else:   # Draw all boxes using same BGRcolor
            corners2d = self.getBoxCorners2d(nusc_det)
            self.draw_bboxes(corners2d, BGRcolor, thickness, **kwargs)
            if draw_vel:
                self._draw_vel(nusc_det, BGRcolor, thickness, **kwargs)
            if draw_id:
                self._draw_id(nusc_det, BGRcolor, **kwargs)

    def _draw_vel(self, nusc_det: list, BGRcolor=(255, 255, 255), thickness=1, alpha=1.0, **kwargs):
        if alpha == 1.0:
            image = self.image
        else:
            image = np.zeros_like(self.image)
        for det in nusc_det:
            vel = det['velocity'][:2]
            if np.linalg.norm(vel) < 0.2:
                continue
            start_point = det['translation'][:2]
            end_point = start_point + vel
            start_point = np.round(start_point).astype(int)
            end_point = np.round(end_point).astype(int)
            image = cv2.arrowedLine(image, start_point, end_point, BGRcolor, thickness)
        if alpha != 1.0:
            self.image = cv2.addWeighted(self.image, 1, image, alpha, 0)

    def _draw_id(self, nusc_det: list, BGRcolor=(255, 255, 255), fontScale=0.8, thickness=1, alpha=1.0, **kwargs):
        if alpha == 1.0:
            image = self.image
        else:
            image = np.zeros_like(self.image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontScale
        for det in nusc_det:
            org = det['translation'][:2]
            org = np.round(org).astype(int)
            ID = int(det['tracking_id'])
            image = cv2.putText(image, str(ID), org, font, fontScale, BGRcolor, thickness, cv2.LINE_AA)
        if alpha != 1.0:
            self.image = cv2.addWeighted(self.image, 1, image, alpha, 0)

    def _draw_grid(self, img, grid_shape, color=(0, 255, 0), thickness=1):
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        return img

    def draw_grid(self, image, diff=10, color=(0, 255, 0), thickness=1, alpha=1.0):
        """ Draw grid from image center """
        h, w, _ = image.shape
        color = np.array(color) * alpha

        # draw vertical lines
        x = w // 2
        while(x < w):
            cv2.line(image, (x, 0), (x, h), color=color, thickness=thickness)
            cv2.line(image, (w-x, 0), (w-x, h), color=color, thickness=thickness)
            x += int(round((diff / self.resolution)))
        
        # draw horizontal lines
        y = h // 2
        while(y < h):
            cv2.line(image, (0, y), (w, y), color=color, thickness=thickness)
            cv2.line(image, (0, h-y), (w, h-y), color=color, thickness=thickness)
            y += int(round((diff / self.resolution)))

        return image

    def draw_img_boundary(self, BGRcolor=(255, 255, 255), thickness=4):
        x, y, w, h = 0, 0, self.width, self.height
        cv2.rectangle(self.image, (x, y), (x+w, y+h), BGRcolor, thickness)
        
    def show(self):
        """
        show and reset the image
        """
        if self.grid:
            # grid_image = np.ones_like(self.image, dtype=np.uint8) * self.background_color
            grid_image = np.ones_like(self.image, dtype=np.uint8)
            grid_image = self.draw_grid(grid_image, diff=10, color=(255, 255, 255), thickness=2, alpha=0.3)
            grid_image = self.draw_grid(grid_image, diff=50, color=(0, 0, 255), thickness=3, alpha=0.5)
            # mask = (self.image == self.background_color)
            # self.image = self.image * np.bitwise_not(mask) + grid_image * mask
            self.image = cv2.addWeighted(grid_image, 0.5, self.image, 1.0, 0)
        self.draw_img_boundary()
        cv2.imshow(self.windowName, self.image)
        self.reset()

def getColorFromID(baseColor=(100, 100, 100), colorRange=(155, 255), ID=-1) -> tuple:
    if ID == -1:  # id == -1
        B, G, R = baseColor # Gray color
    else:
        B = ((25 + 50*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]     # colorRange[0]~colorRange[1]
        G = ((50 + 30*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]     # colorRange[0]~colorRange[1]
        R = ((100 + 20*ID) % (255 - colorRange[0]) + colorRange[0]) % colorRange[1]    # colorRange[0]~colorRange[1]
    return (B, G, R)

def getColorFromID_HSV(baseColor=(100, 100, 100), ID=-1, cycle_num=12):
    # Generate colors using HSV color space
    if ID == -1:
        return baseColor
    else:
        hue = (ID % cycle_num) / cycle_num
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        bgr_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # Convert RGB to BGR
        return bgr_color

if __name__ == "__main__":
    trackViz = TrackVisualizer()
    trackViz.draw_radar_pts([[100, 200+i*10], [200, 300]], {'translation':[0, 0, 0, 0]})
    