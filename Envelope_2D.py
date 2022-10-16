import sys
import configparser as cp
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
import math
from datetime import datetime


def angle_fun(x,y,i,jump_flag):
    if jump_flag:
        alpha = np.arctan2((y[i+1]-y[i-1]),(x[i+1]-x[i-1]))
        beta = np.arctan2((y[i+2]-y[i+1]),(x[i+2]-x[i+1]))
        return abs(alpha-beta)*180/np.pi
    else:
        alpha = np.arctan2((y[i+1]-y[i]),(x[i+1]-x[i]))
        beta = np.arctan2((y[i+2]-y[i+1]),(x[i+2]-x[i+1]))
        return abs(alpha-beta)*180/np.pi




############## Read configuration file
if len(sys.argv) < 2:
    cfg_file = "defaultcfg2D.cfg"
else:
    cfg_file = sys.argv[1]

config = cp.ConfigParser()
config.read(cfg_file)

############## Constants definition, see configuration file for contants description
m = float(config['config']['m'])
rp = float(config['config']['rp'])
helix_angle =  float(config['config']['helix_angle'])
rack_height = float(config['config']['rack_height'])
rack_thickness = float(config['config']['rack_thickness'])
psi = float(config['config']['psi'])
beta = float(config['config']['beta'])
c = float(config['config']['c'])
rack_ratio = float(config['config']['rack_ratio'])
delta_ratio = float(config['config']['delta_ratio'])
theta_trans = float(config['config']['theta_trans'])
deg_step = float(config['config']['deg_step'])
ext_angle = float(config['config']['ext_angle'])
output_filename = config['config']['output_filename']
input_filename = config['config']['input_filename']
date_str = datetime.now().strftime("%y%m%d_%H%M")

d = rp-m+rack_height/2+c*m        #[mm] distance between pinion axes and rack body midline
rack_half_length = ext_angle*(rack_ratio+delta_ratio)+rp

############## Arrays definition

# Steering wheel angles array in [deg], this influences the resolution of the movement and the run-time 
theta_wheel = np.arange(-ext_angle,ext_angle,deg_step)

# Number of discretizations of the movement
N = len(theta_wheel)

# Inizialization of the pinion angles array
theta_pinion = np.empty(N)

# This function define the angular position of the second shaft as function of angular position of the first shaft, the phase angle of the joint and the inclination angle
def cardan_fun(theta,beta,psi):
    # theta is the angular position of the input shaft
    # beta is the phase angle of the joint, when theta is equal to zero beta is the angle between the hinge axis of the first shaft and the shafts plane, it is usually set to 0 or 90 deg
    # psi is the joint angle, in other words it is the acute angle between the first and second shaft's axis 
    
    # conversion of the three angles in [rad]
    t = theta*np.pi/180
    b = beta*np.pi/180
    p = psi*np.pi/180
    # Correction of the points of discontinuity of the function
    if not np.mod(theta-beta+90,180):
        return theta
    # The function of the cardan joint with phase angle beta and inclination angle psi
    else:
        return (np.arctan(np.tan(t-b)/np.cos(p)) + np.arctan(np.tan(b)/np.cos(p)) + np.pi*np.floor((t-b)/np.pi+0.5))*180/np.pi

theta_pinion = [cardan_fun(theta,beta,psi) for theta in theta_wheel]

# Definition of the rack displacement array with constant transmission ratio as function of steering wheel angle
rack_disp = [t*(rack_ratio+delta_ratio)+0.584*delta_ratio*theta_trans*math.erf(-1.5174*t/theta_trans) for t in theta_wheel]

############## CAD objects definition
pinion_points = []
with open('ir329065_punta.xyz') as pinion_file:
    lines = pinion_file.readlines()
    for line in lines:
        x, y = line.strip().split(',')
        pinion_points.append([float(x),float(y)])

pinion = Polygon(pinion_points)

rack = Polygon([[-rack_half_length,0],[rack_half_length,0],[rack_half_length,rack_height],[-rack_half_length,rack_height]])

plane_height = np.linspace(-rack_thickness/2, rack_thickness/2, 10, endpoint=True)

slices = []
fig = plt.figure()
ax_3D = fig.add_subplot(projection='3d')
ax_3D.axis('equal')
fig = plt.figure()
ax_2D = fig.add_subplot()#projection='3d'
ax_2D.axis('equal')

for  j,  height in enumerate(plane_height):
    section_rotation = height/rp*np.sin(helix_angle)
    rack = Polygon([[-rack_half_length,0],[rack_half_length,0],[rack_half_length,rack_height],[-rack_half_length,rack_height]])

    k = 1
    # For each itaretion the pinion is moved in the right position and cut off the rack
    for i in range(N):

        if i>k*N/10:
            print(10*k,'%\n')
            k=k+1

        theta = theta_pinion[i]*np.pi/180 - section_rotation
        rack_ratio_mm_rad = rack_ratio*180/np.pi
        affinity_matrix = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta),rack_disp[i],-rack_ratio_mm_rad]
        rack = rack.difference(affinity.affine_transform(pinion,affinity_matrix))
    
    rack_x, rack_y = rack.exterior.xy
    
    # I find the index of the first corner in order to remove the unnecessary points
    ind = rack_x.index(-rack_half_length)
    if rack_y[ind]!=0:
        if rack_y[ind+1]==0:
            ind = ind+1
        else:
            ind = ind-1

    # I check if the order of the array is correct by checking if the next point is on the plane y = 0, and in case it is not I flip the index of the array
    if rack_y[ind+1]!=0: 
        rack_x = rack_x[::-1]
        rack_y = rack_y[::-1]

    # I erase the 4 corners of the rack slice, keeping only the teeth
    rack_x = rack_x[-ind:] + rack_x[:-ind-4]
    rack_y = rack_y[-ind:] + rack_y[:-ind-4]

    if j == 0:
        ax_2D.plot(rack_x,rack_y,'.')
    
    slice = []
    end_of_curve_index = 0
    i = 0
    final_i = len(rack_x)-1
    while i<final_i:
        curve_x = []
        curve_y = []
        curve_x.extend(rack_x[i:i+2])
        curve_y.extend(rack_y[i:i+2])
        end_of_curve_index = end_of_curve_index+rack_y[end_of_curve_index+2:].index(0)+2
        fillet_flag = 0
        jump_flag = 0
        while i<end_of_curve_index-2:
            angle_bw_segments = angle_fun(rack_x,rack_y,i,jump_flag)
            jump_flag = 0
            if angle_bw_segments>10:
                curve_x.append(rack_x[i+3])
                curve_y.append(rack_y[i+3])
                jump_flag = 1
                fillet_flag = 1
                i += 2
            elif fillet_flag == 1:
                curve_x.extend(rack_x[i+1:end_of_curve_index-1])
                curve_y.extend(rack_y[i+1:end_of_curve_index-1])
                i = end_of_curve_index-2
            else:
                curve_x.append(rack_x[i+2])
                curve_y.append(rack_y[i+2])
                i += 1
                
        if i == end_of_curve_index-1:
            i +=2
        else:    
            curve_x.append(rack_x[end_of_curve_index])
            curve_y.append(rack_y[end_of_curve_index])
            i +=3
        slice.append([curve_x,curve_y])
    slices.append(slice)


for i in range(len(slices)):
    for j in range(len(slices[i])):
        ax_3D.plot(slices[i][j][0],slices[i][j][1],plane_height[i],'b')
        if i ==0:
            ax_2D.plot(slices[0][j][0],slices[0][j][1],'b')

plt.show()
