#!/usr/bin/env python3
import os
import argparse

import cv2
import csv
import numpy as np
import pandas as pd

from scipy import interpolate

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DataExtractor:
    def __init__(self, args):        
        image_topic = '/d435i/color/image_raw'
        depth_topic = "/d435i/aligned_depth_to_color/image_raw"
        odom_topic = "/vins_node/odometry"
        
        bridge = CvBridge()

        print("Extract images from", args.rosbag, "on topic", image_topic)
    
        bag = rosbag.Bag(args.rosbag, "r")
        count = 0
        t_image = 0
        x_path = np.array([])
        y_path = np.array([])
        z_path = np.array([])
        w_hist = np.array([])
        x_hist = np.array([])
        y_hist = np.array([])
        z_hist = np.array([])
        theta_hist = np.array([])
        
        cv_img = np.array([])
        
        # Split bags name to use as frame name
        bag_name = args.rosbag.split('/')
        bag_name = bag_name[-1].split('.')[0]
        
        for topic, msg, t in bag.read_messages(topics=[image_topic, depth_topic, odom_topic]):
            print('topic:', topic)
            if topic == image_topic and (count % 100) < 0.001:
                print('cv_img.size:', cv_img.size)
                if cv_img.size > 0 and x_path.size > 0:
                    # Before reading the new image, let's use the last one to get the patches
                    theta = theta_hist[0] - np.pi/2
                    x_path -= x_path[0]
                    y_path -= y_path[0]
                    z_path -= (z_path[0] + 0.4)
                    #rot_x = x_path*np.cos(theta) - y_path*np.sin(theta)
                    #rot_y = x_path*np.sin(theta) + y_path*np.cos(theta)
                    
                    print('w_hist size:', w_hist.size)
                    print('x_hist size:', x_hist.size)
                    print('y_hist size:', y_hist.size)
                    print('z_hist size:', z_hist.size)
                    #r1 = R.from_euler('z', -90, degrees=True)
                    r = R.from_quat([w_hist[0], x_hist[0], y_hist[0], z_hist[0]])
                    path = np.array([[x_path, y_path, z_path]]).T
                    #path = np.dot(r1.as_dcm(), path)
                    #path = np.dot(r.as_dcm(), path[:,:,0])
                    path = np.dot(r.inv().as_dcm(), path)
                    
                    # convert the path to pixel projections
                    x, y = self.convert_to_pixels(path[0], path[1], path[2])
                    #x, y = self.convert_to_pixels(rot_x, rot_y, z_path)
                    #print('x:', x.T)
                    #print('y:', y.T)
                    
                    #x = np.array_split(x, int(x.size/32))
                    #y = np.array_split(y, int(y.size/32))
                    #cv2.imwrite(os.path.join(args.image_dir, image_name), cv_img)
                    #cv2.imwrite(os.path.join(args.image_dir, depth_name), cv_img.astype(np.uint16))
                    new_img = cv_img.copy()
                    idx = (x >= 0) * (x < 640) * (y >= 0) * (y < 480)
                    x = x[idx]
                    y = y[idx]
                    for i in range(x.size):
                        init_x = max(0,x[i]-16)
                        init_x = min(init_x+32, 639)
                        init_y = max(0,y[i]-16)
                        init_y = min(init_y+32, 478)
                        new_img[init_y:init_y+32,init_x:init_x+32,0] = 255
                    
                    # plot
                    plt.imshow(new_img)
                    plt.draw()
                    plt.pause(1.0)
                    plt.clf()
                
                # Now we can get the new image
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # RGB image
                image_name = "frame_" + bag_name + "_" + str(t.to_nsec()) + ".tif"
                
                t_image = t
                # Clear path arrays
                x_path = np.array([])
                y_path = np.array([])
                z_path = np.array([])
                w_hist = np.array([])
                x_hist = np.array([])
                y_hist = np.array([])
                z_hist = np.array([])
                theta_hist = np.array([])
                
            if topic == depth_topic and t == t_image:
                cv_img = bridge.imgmsg_to_cv2(msg)
                # Depth image
                depth_name = "depth_" + bag_name + "_" + str(t.to_nsec()) + ".tif"
                        
            elif topic == odom_topic:
                x_path = np.append(x_path, msg.pose.pose.position.x)
                y_path = np.append(y_path, msg.pose.pose.position.y)
                z_path = np.append(z_path, msg.pose.pose.position.z)
                
                w = msg.pose.pose.orientation.w
                x = msg.pose.pose.orientation.x
                y = msg.pose.pose.orientation.y
                z = msg.pose.pose.orientation.z
                roll, pitch, theta = self.quaternion_to_euler(w, x, y, z)
                print('roll:', roll, 'pitch:', pitch, 'yaw:', theta)
                w_hist = np.append(w_hist, w)
                x_hist = np.append(x_hist, x)
                y_hist = np.append(y_hist, y)
                z_hist = np.append(z_hist, z)
                theta_hist = np.append(theta_hist, theta)
                
            if topic == image_topic:
                count += 1
            
        # Close rosbag file
        bag.close()        
        
    def constraintAngle(self, angle):
        # reduce the angle  
        constAngle =  angle % (2*np.pi)
        
        # force it to be the positive remainder, so that 0 <= angle < 360  
        constAngle = (constAngle + 2*np.pi) % (2*np.pi)
        
        return constAngle
        
    def create_path(self, path_x, path_y):
        path_x_grid = -path_x/self.map_resolution + self.grid_size[0]/2
        path_y_grid = -path_y/self.map_resolution + self.grid_size[1]/2
        
        indexes = (path_x_grid >= 0) * (path_x_grid < self.grid_size[0]) * \
                  (path_y_grid >= 0) * (path_y_grid < self.grid_size[1])
        
        path_x_grid = path_x_grid[indexes].astype(int)
        path_y_grid = path_y_grid[indexes].astype(int)
        
        return path_x_grid, path_y_grid
    
    def quaternion_to_euler(self, w, x, y, z):
        ysqr = y * y
    
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
    
        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        Y = np.arcsin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)
    
        return X, Y, Z
    
    def convert_to_pixels(self, X, Y, Z):
        #print('X:', X)
        #print('Y:', Y)
        #print('Z:', Z)
        temp = Z
        Z = X
        X = -Y
        Y = -temp        
        # Pinhole camera
        f = 1.93*10**-3
        x = f/Z*X
        y = f/Z*Y
        z = f/Z*Z
        xc = np.array([[x, y, z]]).T
        #print('x:', x)
        #print('y:', y)
        #print('z:', z)
        # Calibration Matrix
        sx = 1.0/640
        sy = 1.0/480
        '''
        Min = np.array([[f*sx, 0,    ox],
                        [0,    f&sy, oy],
                        [0,    0,    1]])
        '''
        Min = np.array([[612.547119140625, 0.0, 319.91510009765625],
                        [0.0, 611.4494018554688, 236.07823181152344],
                        [0.0, 0.0, 1.0]])
        p = 1/f*np.dot(Min, xc)
        #print('p[0]:', p[0,:])
        x = p[0,:].astype(int)
        y = p[1,:].astype(int)
        return x, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", default='bags_list.txt', help='Text file with list of rosbags path')
    parser.add_argument("--image_dir", help="Image output directory.")
    parser.add_argument("--map_resolution", default=0.1, help='Resolution of each pixel in the grid map')
    parser.add_argument("--csv_name", help='Output CSV file name')
    
    args = parser.parse_args()
    
    data_obj = DataExtractor(args)
    #data = data_obj.get_data()