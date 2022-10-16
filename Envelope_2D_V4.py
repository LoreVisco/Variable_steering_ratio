#################################################################################### Header                     #region
from asyncio import ensure_future
from operator import truediv
import sys
import copy
import configparser as cp
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
import cadquery as cq
from colorama import init
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
import math
from datetime import datetime

#endregion
#################################################################################### Functions definition       #region
def delta_1_joint(delta_2,beta,alpha):
    # theta is the angular position of the input shaft
    # beta is the phase angle of the joint, when theta is equal to zero beta is the angle between the hinge axis of the first shaft and the shafts plane, it is usually set to 0 or 90 deg
    # alpha is the joint angle, in other words it is the acute angle between the first and second shaft's axis 
    
    # conversion of the three angles in [rad]
    d = delta_2*np.pi/180
    b = beta*np.pi/180
    a = alpha*np.pi/180

    # Correction of the points of discontinuity of the function
    correction = np.pi*np.ceil((d-np.pi/2+np.arctan2(np.tan(b),np.cos(a)))/np.pi)

    delta_1 = np.arctan(np.cos(a)*np.tan(d+np.arctan2(np.tan(b),np.cos(a))))-b+correction

    return delta_1*180/np.pi

def deriv(A,B):
    length = min([len(A),len(B)])
    C = np.empty(length)
    for i in range(length-1):
        C[i] = (A[i+1]-A[i])/(B[i+1]-B[i])
    C[i+1] = C[i]+(B[i+1]-B[i])*(C[i]-C[i-1])/(B[i]-B[i-1])
    return C

# This function define the angular position of the second shaft as function of angular position of the first shaft, the phase angle of the joint and the inclination angle
def delta_2_joint(delta_1,beta,alpha):

    d = delta_1*np.pi/180
    b = beta*np.pi/180
    a = alpha*np.pi/180

    correction = np.pi*np.ceil((d-np.pi/2+b)/np.pi)
    
    delta_2 = np.arctan2(np.tan(d+b),np.cos(a))-np.arctan2(np.tan(b),np.cos(a))+correction
       
    return delta_2*180/np.pi

def variable_ratio_fun(delta,initial_ratio,Delta,delta_transition):
    displacement = (initial_ratio+Delta)*delta - Delta*delta_transition/2*np.sqrt(np.pi/3)*math.erf(delta/delta_transition*np.sqrt(3))
    return displacement

def variable_ratio_inverse(displacement,initial_ratio,Delta,delta_transition):
    delta = displacement/(initial_ratio+Delta)
    err = 1
    while abs(err)>1e-6:
        delta_old = delta
        delta = (displacement + Delta*delta_transition/2*np.sqrt(np.pi/3)*math.erf(np.sqrt(3)*delta_old/delta_transition))/(initial_ratio + Delta)
        err = abs((delta-delta_old)/delta)
    return delta

def erase_next(x,y,i,start):

    alpha_1 = np.arctan2((y[i-1]-y[i-2]),(x[i-1]-x[i-2]))*180/np.pi
    alpha = np.arctan2((y[i]-y[i-1]),(x[i]-x[i-1]))*180/np.pi
    beta_1 = np.arctan2((y[i+1]-y[i]),(x[i+1]-x[i]))*180/np.pi
    beta_2 = np.arctan2((y[i+2]-y[i]),(x[i+2]-x[i]))*180/np.pi
    distance = np.linalg.norm([x[i+1]-x[i],y[i+1]-y[i]])
    d_min = 0.075
    d_max = 0.2
    max_diff = 1#+abs(alpha-alpha_1)/10
    Fillet_min_diff = 1

    if distance<d_min:
        return True
    if i>start+1:
        if abs((alpha-alpha_1)-(beta_1-alpha))<max_diff:
            return False
        elif (beta_1-alpha)<0:
            if (beta_2-alpha)-(beta_1-alpha)>Fillet_min_diff:
                return True
            else:
                return False
        elif distance>d_max:
            return False
        elif beta_1-alpha>=0:
            return False
    else:
        return False
#endregion
#################################################################################### Read configuration file    #region
if len(sys.argv) < 2:
    cfg_file = "defaultcfg2D.cfg"
else:
    cfg_file = sys.argv[1]

config = cp.ConfigParser()
config.read(cfg_file)

# see configuration file for contants description
mn = float(config['config']['mn'])
rp = float(config['config']['rp'])
helix_angle =  float(config['config']['helix_angle'])
rack_height = float(config['config']['rack_height'])
alpha = float(config['config']['alpha'])
beta = float(config['config']['beta'])
c = float(config['config']['c'])
rack_stroke = float(config['config']['rack_stroke'])
rack_ratio = float(config['config']['rack_ratio'])
Delta_ratio = float(config['config']['Delta_ratio'])
delta_trans = float(config['config']['delta_trans'])
deg_step = float(config['config']['deg_step'])
height_discretizations = int(config['config']['height_discretizations'])
output_filename = config['config']['output_filename']
pinion_filename = config['config']['pinion_filename']

#endregion
#################################################################################### Read pinion input file     #region
# Import of the pinion section from a .xyz file, each line of this file contain the coordinates of the points of the boundary of the section
pinion_points = []
with open(pinion_filename) as pinion_file:
    lines = pinion_file.readlines()
    # For each line I extract the x and y coordinates and assign them to the variable pinion_points
    for line in lines:
        x, y = line.strip().split(',')
        pinion_points.append([float(x),float(y)])
#endregion
#################################################################################### Constants definition       #region

date_str = datetime.now().strftime("%y%m%d_%H%M_")

find_tip_radius = [np.linalg.norm(pts) for pts in pinion_points]
tip_radius = max(find_tip_radius)
tip_index = find_tip_radius.index(tip_radius)
initial_rotation = np.pi/2-np.arctan2(pinion_points[tip_index][1],pinion_points[tip_index][0])

m = mn/np.cos(helix_angle*np.pi/180)
extreme_steering_wheel_angle = variable_ratio_inverse(rack_stroke,rack_ratio,Delta_ratio,delta_trans)
z = round(2*rp/m)
angle_bw_teeth = 360/z
ext_angle = (math.ceil(delta_2_joint(extreme_steering_wheel_angle,beta,alpha)/angle_bw_teeth+0.5)+1)*angle_bw_teeth
axis_distance = rp-m*(1-c)
delta_pinion_tip_exit = np.arccos(axis_distance/tip_radius)*180/np.pi
delta_pinion_tooth_completion = rack_height/2/rp*np.tan(helix_angle*np.pi/180)*180/np.pi
delta_pinion_overtravel = delta_pinion_tip_exit + delta_pinion_tooth_completion
l_right = variable_ratio_fun(ext_angle+delta_pinion_overtravel,rack_ratio,Delta_ratio,delta_trans)+tip_radius
l_left = variable_ratio_fun(delta_pinion_overtravel,rack_ratio,Delta_ratio,delta_trans)+tip_radius
rack_thickness = tip_radius-axis_distance+1

#endregion
#################################################################################### Arrays definition          #region
# Steering wheel angles array in [deg], this influences the resolution of the movement and the run-time 
delta_pinion = np.arange(-delta_pinion_overtravel,ext_angle+delta_pinion_overtravel,deg_step)

# Number of discretizations of the movement
N = len(delta_pinion)

delta_wheel = [delta_1_joint(d,beta,alpha) for d in delta_pinion]
# delta_pinion = delta_wheel # Activate this line if you want to deactivate cardanic joint correction

# Definition of the rack displacement array with constant transmission ratio as function of steering wheel angle
rack_disp = [variable_ratio_fun(d,rack_ratio,Delta_ratio,delta_trans) for d in delta_wheel]

# This is the vector of the heights of the planes on wich we want to find the profile of the teeth, the number of planes will affect the accuracy of the result
plane_height = np.linspace(-rack_height/2, rack_height/2, height_discretizations, endpoint=True)
#endregion
#################################################################################### Slicing/Smoothing          #region
# I create the two polygons, the pinion from the imported points and the rack to be cut is just a rectangle 
pinion = affinity.rotate(Polygon(pinion_points),initial_rotation,origin=(0,0),use_radians=True)

#### Uncomment the following to check on pinion geometry
# fig = plt.figure()
# pinion_ax = fig.add_subplot()
# pinion_ax.axis('equal')
# pinion_x, pinion_y = pinion.exterior.xy
# plt.plot(pinion_x,pinion_y)
# plt.show()

# Initialization of the two arrays output of the following for cycle
slices = []
slices_raw = []

# In this for cycles through the different slices of the rack, first the rack profile is computed using the boolean operations,
# in order to address the helixicity of the teeth for each slice the pinion section is rotated accordingly, after the realization of the polygon
# representing the slice of the rack, the points are extracted, the four points on the edges of the rack are eliminated.
# Then, the array is ordered, the fillet on each tooth is smoothed out and the array is divided into curves.
for  height in plane_height:
    section_rotation = height/rp*np.sin(helix_angle)
    rack = Polygon([[-l_left,0],[l_right,0],[l_right,rack_thickness],[-l_left,rack_thickness]])

    first_midpoint_x = variable_ratio_fun(delta_1_joint(-height/rp*np.tan(helix_angle)*180/np.pi,beta,alpha),rack_ratio,Delta_ratio,delta_trans)
    last_midpoint_x = variable_ratio_fun(delta_1_joint(ext_angle-height/rp*np.tan(helix_angle)*180/np.pi,beta,alpha),rack_ratio,Delta_ratio,delta_trans)

    k = 1
    # For each itaretion the pinion is moved in the right position and cut off the rack
    for i in range(N):

        if i>k*N/10:
            print(10*k,'%\n')
            k=k+1

        theta = delta_pinion[i]*np.pi/180 - section_rotation
        affinity_matrix = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta),rack_disp[i],-axis_distance]
        rack = rack.difference(affinity.affine_transform(pinion,affinity_matrix))
    
    rack_x, rack_y = rack.exterior.xy
    
    # I find the index of the first corner in order to remove the unnecessary points
    ind = rack_x.index(-l_left)
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
    
    x_raw = copy.deepcopy(rack_x)
    y_raw = copy.deepcopy(rack_y)
    slices_raw.append([x_raw,y_raw])
    
    slice = []

    start_of_curve_index = (np.abs(rack_x-first_midpoint_x)).argmin()
    while rack_y[start_of_curve_index]!=0:
        start_of_curve_index -= 1

    i = start_of_curve_index
    while rack_x[i]<last_midpoint_x:
        i += 1
        end_of_curve_index = i
        while rack_y[end_of_curve_index]!=0:
            end_of_curve_index += 1
        while i<end_of_curve_index-1:
            if erase_next(rack_x,rack_y,i,start_of_curve_index):
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

#endregion
#################################################################################### Plots                      #region
# Inizialization of the figures used check the result and to debug the code

fig = plt.figure()
ax_3D = fig.add_subplot(projection='3d')
ax_3D.axis('equal')
fig = plt.figure()
ax_2D = fig.add_subplot()
ax_2D.axis()#'equal'

slice_to_check = 13

delta_wheel_ratio = [1/d for d in deriv(delta_wheel,rack_disp)]
delta_pinion_ratio = [1/d for d in deriv(delta_pinion,rack_disp)]
ax_2D.plot(rack_disp,delta_wheel_ratio)
ax_2D.plot(rack_disp,delta_pinion_ratio)
for i in range(len(slices)):
    if i == slice_to_check:
        ax_2D.plot(slices_raw[i][0],slices_raw[i][1],'.r')
    for j in range(len(slices[i])):
        ax_3D.plot(slices[i][j][0],slices[i][j][1],plane_height[i],'b')
        if i == slice_to_check:
            ax_2D.plot(slices[i][j][0],slices[i][j][1],'b')
            ax_2D.plot(slices[i][j][0],slices[i][j][1],'og')

ax_2D.grid(True)
plt.show()
#endregion
#################################################################################### Spline/STEP file creation  #region

result = cq.Workplane('XY')
for k in range(len(slices)):

    for j in range(len(slices[k])):
        MSpline_vec = []
        spline_curve = []

        for i in range(len(slices[k][j][0])):
            MSpline_vec.append(cq.Vector(slices[k][j][0][i],slices[k][j][1][i],plane_height[k]))
        result.objects.append(cq.Edge.makeSpline(MSpline_vec))

for j in range(len(slices[0])):

    MSpline_vec_left = []
    MSpline_vec_right = []

    for k in range(len(slices)):
        MSpline_vec_left.append(cq.Vector(slices[k][j][0][0],slices[k][j][1][0],plane_height[k]))
        MSpline_vec_right.append(cq.Vector(slices[k][j][0][-1],slices[k][j][1][-1],plane_height[k]))

    result.objects.append(cq.Edge.makeSpline(MSpline_vec_left))
    result.objects.append(cq.Edge.makeSpline(MSpline_vec_right))

cq.exporters.export(result,date_str+output_filename+'.STEP')

#endregion