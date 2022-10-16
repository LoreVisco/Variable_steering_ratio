from asyncio import ensure_future
from operator import truediv
import sys
import configparser as cp
import cadquery as cq
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon

import math
from datetime import datetime


def erase_next(x,y,i):

    alpha = np.arctan2((y[i]-y[i-1]),(x[i]-x[i-1]))
    beta_1 = np.arctan2((y[i+1]-y[i]),(x[i+1]-x[i]))
    beta_2 = np.arctan2((y[i+2]-y[i]),(x[i+2]-x[i]))
    if abs(beta_1-alpha)*180/np.pi>30:
        return True
    elif abs(beta_1-alpha)*180/np.pi<3:
        return False
    if beta_2-alpha>=beta_1-alpha:
        return True
    else:
        return False

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
rack_thickness = float(config['config']['rack_thickness'])
rack_height = float(config['config']['rack_height'])
alpha = float(config['config']['alpha'])
beta = float(config['config']['beta'])
c = float(config['config']['c'])
rack_ratio = float(config['config']['rack_ratio'])
Delta_ratio = float(config['config']['Delta_ratio'])
delta_trans = float(config['config']['delta_trans'])
deg_step = float(config['config']['deg_step'])
ext_angle = float(config['config']['ext_angle'])
output_filename = config['config']['output_filename']
pinion_filename = config['config']['pinion_filename']
date_str = datetime.now().strftime("%y%m%d_%H%M")

d = rp-m+rack_thickness/2+c*m        #[mm] distance between pinion axes and rack body midline
rack_half_length = ext_angle*(rack_ratio+Delta_ratio)+rp

#################################################################################### Arrays definition

# Steering wheel angles array in [deg], this influences the resolution of the movement and the run-time 
delta_wheel = np.arange(-ext_angle,ext_angle,deg_step)

# Number of discretizations of the movement
N = len(delta_wheel)

# Inizialization of the pinion angles array
delta_pinion = np.empty(N)

# This function define the angular position of the second shaft as function of angular position of the first shaft, the phase angle of the joint and the inclination angle
def cardan_fun(theta,beta,alpha):
    # theta is the angular position of the input shaft
    # beta is the phase angle of the joint, when theta is equal to zero beta is the angle between the hinge axis of the first shaft and the shafts plane, it is usually set to 0 or 90 deg
    # alpha is the joint angle, in other words it is the acute angle between the first and second shaft's axis 
    
    # conversion of the three angles in [rad]
    t = theta*np.pi/180
    b = beta*np.pi/180
    a = alpha*np.pi/180
    # Correction of the points of discontinuity of the function
    if not np.mod(theta-beta+90,180): # True only for theta-beta is equal to [ ..., -90, 90, 270, ...]
        return theta
    # The function of the cardan joint with phase angle beta and inclination angle alpha
    else:
        return (np.arctan2(np.tan(t-b),np.cos(a)) + np.arctan2(np.tan(b),np.cos(a)) + np.pi*np.floor((t-b)/np.pi+0.5))*180/np.pi

# delta_pinion = [cardan_fun(theta,beta,alpha) for theta in delta_wheel]
delta_pinion = delta_wheel # Activate this line if you want to deactivate cardanic joint correction

# Definition of the rack displacement array with constant transmission ratio as function of steering wheel angle
rack_disp = [t*(rack_ratio+Delta_ratio)+0.584*Delta_ratio*delta_trans*math.erf(-1.5174*t/delta_trans) for t in delta_wheel]
# rack_disp = [t*rack_ratio for t in delta_wheel] #delete this

#################################################################################### CAD objects definition

# Import of the pinion section from a .xyz file, each line of this file contain the coordinates of the points of the boundary of the section
pinion_points = []
with open(pinion_filename) as pinion_file:
    lines = pinion_file.readlines()
    # For each line I extract the x and y coordinates and assign them to the variable pinion_points
    for line in lines:
        x, y = line.strip().split(',')
        pinion_points.append([float(x),float(y)])

find_tip_radius = [np.linalg.norm(pts) for pts in pinion_points]
tip_radius = max(find_tip_radius)
tip_index = find_tip_radius.index(tip_radius)
initial_rotation = np.pi/2-np.arctan2(pinion_points[tip_index][1],pinion_points[tip_index][0])

# I create the two polygons, the pinion from the imported points and the rack to be cut is just a rectangle 
pinion = affinity.rotate(Polygon(pinion_points),initial_rotation,origin=(0,0),use_radians=True)

#### Uncomment the following to check on pinion geometry
# fig = plt.figure()
# pinion_ax = fig.add_subplot()
# pinion_ax.axis('equal')
# pinion_x, pinion_y = pinion.exterior.xy
# plt.plot(pinion_x,pinion_y)
# plt.show()

# This is the vector of the heights of the planes on wich we want to find the profile of the teeth, the number of planes will affect the accuracy of the result
plane_height = np.linspace(-rack_height/2, rack_height/2, 5, endpoint=True)


# Inizialization of the figures used check the result and to debug the code
fig = plt.figure()
ax_3D = fig.add_subplot(projection='3d')
ax_3D.axis('equal')
fig = plt.figure()
ax_2D = fig.add_subplot()#projection='3d'
ax_2D.axis('equal')

# Initialization of the two arrays output of the following for cycle
slices = []
slices_raw = []

# In this for cycles through the different slices of the rack, first the rack profile is computed using the boolean operations,
# in order to address the helixicity of the teeth for each slice the pinion section is rotated accordingly, after the realization of the polygon
# representing the slice of the rack, the points are extracted, the four points on the edges of the rack are eliminated.
# Then, the array is ordered, the fillet on each tooth is smoothed out and the array is divided into curves.
for  height in plane_height:
    section_rotation = height/rp*np.sin(helix_angle)
    rack = Polygon([[-rack_half_length,0],[rack_half_length,0],[rack_half_length,rack_thickness],[-rack_half_length,rack_thickness]])

    k = 1
    # For each itaretion the pinion is moved in the right position and cut off the rack
    for i in range(N):

        if i>k*N/10:
            print(10*k,'%\n')
            k=k+1

        theta = delta_pinion[i]*np.pi/180 - section_rotation
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
    
    x_raw = np.array(rack_x)
    y_raw = np.array(rack_y)
    slices_raw.append([x_raw,y_raw])

    slice = []
    end_of_curve_index = 0
    start_of_curve_index = 0
    i = 0
    while i<len(rack_x)-1:
        i += 1
        end_of_curve_index = end_of_curve_index+rack_y[end_of_curve_index+2:].index(0)+2
        while i<end_of_curve_index-1:
            if erase_next(rack_x,rack_y,i):
                del rack_x[i+1], rack_y[i+1]
                end_of_curve_index -= 1
            else:
                i += 1
        if i==end_of_curve_index:
            i += 1
        else:
            i += 2
        slice.append([rack_x[start_of_curve_index:i],rack_y[start_of_curve_index:i]])
        start_of_curve_index = i      
    slices.append(slice)

####################################################################################

slice_to_check = 2
for i in range(len(slices)):
    slices[i] = slices[i][1:-1]
    for j in range(len(slices[i])):
        ax_3D.plot(slices[i][j][0],slices[i][j][1],plane_height[i],'b')
        if i == slice_to_check:
            ax_2D.plot(slices[i][j][0],slices[i][j][1],'b')
    if i == slice_to_check:
        ax_2D.plot(slices_raw[i][0],slices_raw[i][1],'.r')
plt.show()

####################################################################################

result = cq.Workplane('XY')
for k in range(len(slices)):

    for j in range(len(slices[k])):
        MSpline_vec = []
        spline_curve = []

        for i in range(len(slices[k][j][0])):
            MSpline_vec.append(cq.Vector(slices[k][j][0][i],slices[k][j][1][i],plane_height[k]))
        result.objects.append(cq.Edge.makeSpline(MSpline_vec))
        

####################################################################################

cq.exporters.export(result,'prova_spline.STEP')