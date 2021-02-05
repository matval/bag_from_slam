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
        
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.z_path = np.array([])
        self.w_hist = np.array([])
        self.x_hist = np.array([])
        self.y_hist = np.array([])
        self.z_hist = np.array([])
        self.t_odom = np.array([])
        #self.cv_img = np.array([])
        self.cv_img = np.empty((1, 480, 640, 3), int)
        self.t_img = np.array([])
        self.cv_depth = np.empty((1, 480, 640), int)
        self.t_depth = np.array([])
        #self.theta_hist = np.array([])
        
        cv_img = np.array([])
        cv_depth = np.array([])
        
        # Split bags name to use as frame name
        bag_name = args.rosbag.split('/')
        self.bag_name = bag_name[-1].split('.')[0]
        
        got_image = False
        for topic, msg, t in bag.read_messages(topics=[image_topic, depth_topic, odom_topic]):
            print('topic:', topic)
            t = t.to_nsec()
            if topic == image_topic and (count % 30) < 0.001:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv_img = np.expand_dims(cv_img,axis=0)
                print('cv_img:', cv_img.shape)
                print('self.cv_img:', self.cv_img.shape)
                self.cv_img = np.append(self.cv_img, cv_img, axis=0)
                self.t_img = np.append(self.t_img, t)
                got_image = True
                
            if topic == depth_topic and got_image:
                cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
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
            
        self.cv_img = self.cv_img[1:]
        self.cv_depth = self.cv_depth[1:]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", default='bags_list.txt', help='Text file with list of rosbags path')
    parser.add_argument("--image_dir", help="Image output directory.")
    parser.add_argument("--map_resolution", default=0.1, help='Resolution of each pixel in the grid map')
    parser.add_argument("--csv_name", help='Output CSV file name')
    
    args = parser.parse_args()
    
    data_obj = DataExtractor(args)
    #data = data_obj.get_data()
    count = 0
    
    for idx in range(len(data_obj.t_img)):
        idx_path = bisect.bisect_left(data_obj.t_odom, data_obj.t_img[idx])
        
        num_hor = 100
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
        r1 = R.from_euler('z', 0, degrees=True)
        r = r1 * r.inv()
        
        path = np.array([x_path, y_path, z_path])
        #path = np.dot(r1.as_dcm(), path)
        path = np.dot(r.as_dcm(), path)
        #path = path[:,:,0]
        #path = path - path[:,:1]*np.ones(path.shape)
        
        # correct camera height
        path[1] += 0.4
        
        # convert the path to pixel projections
        x, y = data_obj.convert_to_pixels(path[0], path[1], path[2])
        print('data_obj.cv_img:', data_obj.cv_img.shape)
        new_img = data_obj.cv_img[idx].copy()
        new_depth = data_obj.cv_depth[idx].copy()
        print('new_img:', new_img.shape)
        pix_idx = (x >= 0) * (x < 640) * (y >= 0) * (y < 480)
        x = x[pix_idx]
        y = y[pix_idx]
        
        # Get only unique points
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        temp_c = np.concatenate((x, y), axis=0)
        new_array = [tuple(row) for row in temp_c.T]
        #index = np.unique(temp_c, return_index=True)[1]
        #print('indexes', index)
        #temp_c = np.unique(new_array, axis=0)
        temp_c = pd.unique(new_array)
        temp_c = np.array(temp_c, dtype='int,int')
        #print('unique array ', temp_c['f1'])
        x, y = temp_c['f0'], temp_c['f1']
        
        print_img = new_img.copy()
        for i in range(x.size):
            init_x = max(0,x[i]-16)
            init_x = min(init_x+32, 640-32)
            init_y = max(0,y[i]-16)
            init_y = min(init_y+32, 480-32)
            
            cv_img = new_img[init_y:init_y+32,init_x:init_x+32,:]
            cv_depth = new_depth[init_y:init_y+32,init_x:init_x+32]
            
            print_img[init_y:init_y+32,init_x:init_x+32,0] = 255
            
            image_name = "frame_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
            cv2.imwrite(os.path.join(args.image_dir, image_name), cv_img)
            depth_name = "depth_" + data_obj.bag_name + "_" + '{0:06d}'.format(count) + ".tif"
            cv2.imwrite(os.path.join(args.image_dir, depth_name), cv_depth.astype(np.uint16))
            
            count +=1
        
        # plot
        plt.figure(1)
        #plt.subplot(1, 2, 1)
        plt.imshow(print_img)
        time_diff = np.abs(data_obj.t_odom[idx_path] - data_obj.t_img[idx])/10**9
        plt.title('Time diff: ' + str(time_diff) + ' s')
        '''
        plt.subplot(1, 2, 2)
        plt.plot(data_obj.t_odom)
        plt.plot(idx_path, data_obj.t_odom[idx_path], 'o')
        time_diff = np.abs(data_obj.t_odom[idx_path] - data_obj.t_img[idx])/10**9
        plt.title('Time diff: ' + str(time_diff) + ' s')
        '''
        plt.draw()
        plt.pause(0.5)
        plt.clf()
        #plt.pause(0.01)
        #plt.clf()
        '''
        plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.scatter3D(path[0], path[1], path[2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-10, 10)
        plt.draw()
        plt.pause(0.5)
        plt.clf()
        '''
        
        