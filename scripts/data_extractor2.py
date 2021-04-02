#!/usr/bin/env python3
import os
import argparse

import cv2
import csv
import numpy as np
import pandas as pd
import bisect

from scipy import interpolate

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from mpl_toolkits import mplot3d
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
'''
image_size = (480, 640)

class DataExtractor:
    def __init__(self, args, bag_file):        
        image_topic = '/d435i/color/image_raw'
        depth_topic = "/d435i/aligned_depth_to_color/image_raw"
        odom_topic = "/vins_node/odometry"
        
        bridge = CvBridge()

        # Create buffers
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.z_path = np.array([])
        self.w_hist = np.array([])
        self.x_hist = np.array([])
        self.y_hist = np.array([])
        self.z_hist = np.array([])
        self.t_odom = np.array([])
        #self.cv_img = np.array([])
        self.cv_img = np.empty((1, image_size[0], image_size[1], 3), int)
        self.t_img = np.array([])
        self.cv_depth = np.empty((1, image_size[0], image_size[1]), int)
        self.t_depth = np.array([])
        #self.theta_hist = np.array([])
        
        cv_img = np.array([])
        cv_depth = np.array([])
        
        print("Extract images from", bag_file, "on topic", image_topic)
    
        bag = rosbag.Bag(bag_file, "r")
        count = 0
        t_image = 0
        
        # Split bags name to use as frame name
        bag_name = bag_file.split('/')
        self.bag_name = bag_name[-1].split('.')[0]
        
        got_image = False
        for topic, msg, t in bag.read_messages(topics=[image_topic, depth_topic, odom_topic]):
            print('topic:', topic)
            t = t.to_nsec()
            if topic == image_topic and (count % 30) < 0.001:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_img = cv2.resize(cv_img, (image_size[1], image_size[0]), interpolation = cv2.INTER_CUBIC)
                cv_img = np.expand_dims(cv_img,axis=0)
                print('cv_img:', cv_img.shape)
                print('self.cv_img:', self.cv_img.shape)
                self.cv_img = np.append(self.cv_img, cv_img, axis=0)
                self.t_img = np.append(self.t_img, t)
                got_image = True
                
            if topic == depth_topic and got_image:
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv_depth = cv2.resize(cv_depth, (image_size[1], image_size[0]), interpolation = cv2.INTER_CUBIC)
                cv_depth = np.expand_dims(cv_depth,axis=0)
                print('cv_depth:', cv_depth.shape)
                print('self.cv_depth:', self.cv_depth.shape)
                self.cv_depth = np.append(self.cv_depth, cv_depth, axis=0)
                self.t_depth = np.append(self.t_depth, t)
                got_image = False
                # Depth image
                #depth_name = "depth_" + bag_name + "_" + str(t.to_nsec()) + ".tif"
                        
            elif topic == odom_topic:
                self.x_path = np.append(self.x_path, msg.pose.pose.position.x)
                self.y_path = np.append(self.y_path, msg.pose.pose.position.y)
                self.z_path = np.append(self.z_path, msg.pose.pose.position.z)
                
                w = msg.pose.pose.orientation.w
                x = msg.pose.pose.orientation.x
                y = msg.pose.pose.orientation.y
                z = msg.pose.pose.orientation.z
                #roll, pitch, theta = self.quaternion_to_euler(w, x, y, z)
                #print('roll:', roll, 'pitch:', pitch, 'yaw:', theta)
                self.w_hist = np.append(self.w_hist, w)
                self.x_hist = np.append(self.x_hist, x)
                self.y_hist = np.append(self.y_hist, y)
                self.z_hist = np.append(self.z_hist, z)
                self.t_odom = np.append(self.t_odom, t)
                #theta_hist = np.append(theta_hist, theta)
            
            if topic == image_topic:
                count += 1
            
        # Close rosbag file
        bag.close()
        
        self.cv_img = self.cv_img[1:]
        self.cv_depth = self.cv_depth[1:]
        
    def create_path(self, path_x, path_y):
        path_x_grid = -path_x/self.map_resolution + self.grid_size[0]/2
        path_y_grid = -path_y/self.map_resolution + self.grid_size[1]/2
        
        indexes = (path_x_grid >= 0) * (path_x_grid < self.grid_size[0]) * \
                  (path_y_grid >= 0) * (path_y_grid < self.grid_size[1])
        
        path_x_grid = path_x_grid[indexes].astype(int)
        path_y_grid = path_y_grid[indexes].astype(int)
        
        return path_x_grid, path_y_grid
    
    def convert_to_pixels(self, X, Y, Z):
        #print('X:', X)
        #print('Y:', Y)
        #print('Z:', Z)
        '''
        temp = Z
        Z = X
        X = -Y
        Y = -temp  
        '''
        # Pinhole camera
        #f = 1.93*10**-3
        x = X/Z
        y = Y/Z
        z = Z/Z
        xc = np.array([[x, y, z]]).T
        #xc = np.array([[X, Y, Z]]).T
        #print('x:', x)
        #print('y:', y)
        #print('z:', z)
        # Calibration Matrix
        #sx = 1.0/640
        #sy = 1.0/480
        '''
        Min = np.array([[f*sx, 0,    ox],
                        [0,    f&sy, oy],
                        [0,    0,    1]])
        '''
        Min = np.array([[612.547119140625, 0.0, 319.91510009765625],
                        [0.0, 611.4494018554688, 236.07823181152344],
                        [0.0, 0.0, 1.0]])
        #p = 1/f*np.dot(Min, xc)
        p = np.dot(Min, xc)
        #print('p[0]:', p[0,:])
        u = p[0,:].astype(int)
        v = p[1,:].astype(int)
        return u, v

    def depth_to_gridmap(self, depth, resolution, grid_size, center):
        # Pinhole camera
        #f = 1.93*10**-3
        W = depth.shape[1]
        H = depth.shape[0]
        print('width:', W)
        print('height:', H)
        depth_array = depth.flatten() * 10**-3
        i_coords, j_coords = np.meshgrid(range(H), range(W), indexing='ij')

        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        k_coords = np.ones_like(i_coords)

        p = np.stack((j_coords, i_coords, k_coords))

        Min = np.array([[612.547119140625, 0.0, 319.91510009765625],
                        [0.0, 611.4494018554688, 236.07823181152344],
                        [0.0, 0.0, 1.0]])

        xc = np.linalg.solve(Min, p)

        x = xc * depth_array

        #print("x:", x)
        # Filter the pixels with 0 distance or less
        x = x[:, (x[2] > 0)]

        # Rotate pointcloud x-axis -90 degrees
        T1 = R.from_euler('x', -90, degrees=True)
        T2 = R.from_euler('z', 90, degrees=True)
        T = T2 * T1
        x = np.dot(T.as_dcm(), x)

        # Convert meters to grid coordinates
        x = np.ceil(x/resolution).astype(int)
        x[0] += center[0]
        x[1] += center[1]
        x = x[:, (x[0] >= 0) & (x[1] >= 0) & (x[0] < grid_size[0]) & (x[1] < grid_size[1])]

        grid = np.zeros((max(x[0])+1, max(x[1])+1))
        for i in range(x.shape[1]):
            grid[x[0,i], x[1,i]] += 1

        print("grid:", grid_size)
        new_grid = np.zeros(grid_size)
        new_grid[:grid.shape[0],:grid.shape[1]] = grid
        return new_grid

    def path_to_gridmap(self, path, resolution, grid_size, center, radius):
        # Rotate path to aligned to image
        T1 = R.from_euler('z', 90, degrees=True)
        T2 = R.from_euler('y', -90, degrees=True)
        T = T2 * T1
        x = np.dot(T.as_dcm(), path)

        # Convert meters to grid coordinates
        x = np.ceil(x/resolution).astype(int)
        x[0] += center[0]
        x[1] += center[1]

        x = x[:, (x[0] >= 0) & (x[1] >= 0) & (x[0] < grid_size[0]) & (x[1] < grid_size[1])]

        grid = np.ones(grid_size)

        radius = int(radius/resolution)
        for i in range(x.shape[1]):
            grid[x[0,i]-radius:x[0,i]+radius, x[1,i]-radius:x[1,i]+radius] = 0

        print("path grid:", grid.shape)
        return grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags_list", default='bags_list.txt', help='Text file with list of rosbags path')
    parser.add_argument("--image_dir", help="Image output directory.")
    parser.add_argument("--csv_name", help='Output CSV file name')
    parser.add_argument("--get_freq", default=15, help='Frequency we get the frames from rosbag')
    parser.add_argument("--robot_radius", default=0.15, help='Robot radius to create traversable path')
    parser.add_argument("--horizon", default=200, help='Horizon length in number of points')
    parser.add_argument("--threshold", default=30, help='Threshold value to consider traversable on grid_map')
    parser.add_argument("--map_resolution", default=0.05, help='Resolution in meters for pixels in the gridmap')
    
    args = parser.parse_args()

    rgb_path = os.path.join(args.image_dir, 'rgb')
    depth_path = os.path.join(args.image_dir, 'depth')
    grid_path = os.path.join(args.image_dir, 'gridmap')
    label_path = os.path.join(args.image_dir, 'label')

    # Create directories in case they don't exist
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if not os.path.exists(grid_path):
        os.makedirs(grid_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    with open(os.path.join(args.image_dir, 'data.csv'), 'w') as csvfile:
        fieldnames = ['rgb_img', 'depth_img', 'grid_img', 'label_img']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        #data = data_obj.get_data()
        count = 0
        num_hor = args.horizon
        camera_height = 0.4
        patch_size = (64, 64)

        f = open(args.bags_list, "r")
        lines = f.read().splitlines()
        
        for bag_file in lines:
            data_obj = DataExtractor(args, bag_file)

            for idx in range(len(data_obj.t_img)):
                idx_path = bisect.bisect_left(data_obj.t_odom, data_obj.t_img[idx])
                
                f_idx_path = min(idx_path+num_hor, data_obj.t_odom.size-1)
                
                print('idx_path:', idx_path)
                print('f_idx_path:', f_idx_path)
                
                x_path = data_obj.x_path[idx_path:f_idx_path]
                y_path = data_obj.y_path[idx_path:f_idx_path]
                z_path = data_obj.z_path[idx_path:f_idx_path]
                
                x_path = x_path - x_path[0]
                y_path = y_path - y_path[0]
                z_path = z_path - z_path[0]
                
                w = data_obj.w_hist[idx_path]
                x = data_obj.x_hist[idx_path]
                y = data_obj.y_hist[idx_path]
                z = data_obj.z_hist[idx_path]
                
                r = R.from_quat([x, y, z, w])
                #r1 = R.from_euler('z', 0, degrees=True)
                #r = r1 * r.inv()
                r = r.inv()
                
                path = np.array([x_path, y_path, z_path])
                #path = np.dot(r1.as_dcm(), path)
                path = np.dot(r.as_dcm(), path)
                #path = path[:,:,0]
                #path = path - path[:,:1]*np.ones(path.shape)
                
                # correct camera height
                path[1] += camera_height
                
                print('data_obj.cv_img:', data_obj.cv_img.shape)
                new_img = data_obj.cv_img[idx].copy()
                new_depth = data_obj.cv_depth[idx].copy()

                grid_img = data_obj.depth_to_gridmap(new_depth, 0.01, (200,200), (200,100))
                path_image = data_obj.path_to_gridmap(path, 0.01, (200,200), (200,100), args.robot_radius)

                label_img = 255 * (grid_img > args.threshold) * path_image
                #print('label_img:', label_img.shape)

                # Save label image
                label_name = "label/label_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
                cv2.imwrite(os.path.join(args.image_dir, label_name), label_img)

                # Save depth projection
                grid_name = "grid/grid_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
                cv2.imwrite(os.path.join(args.image_dir, grid_name), grid_img)
                
                # Save raw images
                image_name = "rgb/rgb_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
                cv2.imwrite(os.path.join(args.image_dir, image_name), new_img)
                depth_name = "depth/depth_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
                cv2.imwrite(os.path.join(args.image_dir, depth_name), new_depth.astype(np.uint16))
                
                # Write variables to the document
                writer.writerow({'rgb_img':image_name, 'depth_img':depth_name, 'grid_img':grid_name, 'label_img':label_name})
                
                count +=1      
        