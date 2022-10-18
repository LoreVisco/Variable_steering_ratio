#region#################################################################################### Header                     

import sys
import configparser as cp
from multiprocessing import Pool, cpu_count
import cadquery as cq
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
import math
import tqdm
from datetime import datetime
from scipy import interpolate

#endregion
#region#################################################################################### Functions definition       

def delta_2_joint(delta_1,beta,alpha):
    
    """
    DIRECT FUNCTION OF THE UNIVERSAL JOINT
    
    computes the angle of the output shaft delta_2 for a given configuration of the an universal joint, defined by the input shaft angle delta_1, the joint angle alpha and the phase angle beta. 
    
    ════════════════════════════════
    
    In an universal joint two shafts are involved, input and output shafts, the angular position of the input shaft is called delta_1. Starting from this labeling of the shafts we can define the two angles alpha and beta.
    To do so let's create a fixed frame of reference on the end of the input shaft, the z axis aligned with the shaft axis, the x axis on the plane on which the two shafts lie in the direction of the output shaft and the y axis according with right hand rule.
    Then we create another fixed frame of reference rotated about the z-axis of an angle beta, that frame of reference is the one from which the angle delta_1 is measured, still according with the right-hand rule. 
    beta is called phase angle, in fact the function has a 180 degree periodicity and beta defines the starting point in the period, it is usually set to zero or 90 degree, values at which the transmission ratio (D delta_2)/(D delta_1) has a maximum and a minimum respectively.
    The function is built in such a way that delta_2=0 when delta_1=0, this is because we are often interested just in the difference between the input angle delta_1 and the output angle delta_2.
    the joint angle alpha is the acute angle between the input shaft and the output shaft, it should be positive definite.
    
    ════════════════════════════════
    
    Args:
        delta_1 (float): input shaft angle in degrees
        beta (float): phase angle in degrees in the range (-90,90]
        alpha (float): joint angle in degrees in the range [0,90)

    Returns:
        float: output shaft angle in degrees
    """

    d = delta_1*np.pi/180
    b = beta*np.pi/180
    a = alpha*np.pi/180

    correction = np.pi*np.ceil((d-np.pi/2+b)/np.pi)
    
    delta_2 = np.arctan2(np.tan(d+b),np.cos(a))-np.arctan2(np.tan(b),np.cos(a))+correction
    
    return delta_2*180/np.pi

def delta_1_joint(delta_2,beta,alpha):
    
    """
    INVERSE FUNCTION OF THE UNIVERSAL JOINT
    
    computes the angle of the input shaft delta_1 for a given configuration of the an universal joint, defined by the output shaft angle delta_2, the joint angle alpha and the phase angle beta. 
    
    ════════════════════════════════
    
    In an universal joint two shafts are involved, input and output shafts, the angular position of the input shaft is called delta_1. Starting from this labeling of the shafts we can define the two angles alpha and beta.
    To do so let's create a fixed frame of reference on the end of the input shaft, the z axis aligned with the shaft axis, the x axis on the plane on which the two shafts lie in the direction of the output shaft and the y axis according with right hand rule.
    Then we create another fixed frame of reference rotated about the z-axis of an angle beta, that frame of reference is the one from which the angle delta_1 is measured, still according with the right-hand rule. 
    beta is called phase angle, in fact the function has a 180 degree periodicity and beta defines the starting point in the period, it is usually set to zero or 90 degree, values at which the transmission ratio (D delta_2)/(D delta_1) has a maximum and a minimum respectively.
    The function is built in such a way that delta_2=0 when delta_1=0, this is because we are often interested just in the difference between the input angle delta_1 and the output angle delta_2.
    the joint angle alpha is the acute angle between the input shaft and the output shaft, it should be positive definite.
    
    ════════════════════════════════
    
    Args:
        delta_2 (float): output shaft angle in degrees
        beta (float): phase angle in degrees in the range (-90,90]
        alpha (float): joint angle in degrees in the range [0,90)

    Returns:
        float: input shaft angle in degrees
    """
    
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

def deriv(y,x):
    """
    DERIVATIVE FUNCTION OF Y WITH RESPECT TO X
    
    It computes the numeric derivative of y with respect to x.
    The function takes two array x and y and performs the numerical derivative using the forward difference, the last value is linearly extrapolated from the previous values. It doesn't need x and y to be of the same length, the length of the output will be equal to the length of the shortest.

    ════════════════════════════════
    
    Args:
        y (float): array that defines the function of which to compute the derivative
        x (float): array that defines the variable against which to compute the derivative

    Returns:
        float: array of the same length of the shortest between x and y representing dy/dx
    """
    length = min([len(y),len(x)])
    dy = np.empty(length)
    for i in range(length-1):
        dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    dy[i+1] = dy[i]+(x[i+1]-x[i])*(dy[i]-dy[i-1])/(x[i]-x[i-1])
    return dy

def variable_ratio_fun(delta,initial_ratio,Delta,delta_transition):
    """_summary_

    Args:
        delta (_type_): _description_
        initial_ratio (_type_): _description_
        Delta (_type_): _description_
        delta_transition (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    displacement = (initial_ratio+Delta)*delta - Delta*delta_transition/2*np.sqrt(np.pi/3)*math.erf(delta/delta_transition*np.sqrt(3))
    return displacement

def variable_ratio_inverse(displacement,initial_ratio,Delta,delta_transition):
    """_summary_

    Args:
        displacement (_type_): _description_
        initial_ratio (_type_): _description_
        Delta (_type_): _description_
        delta_transition (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    delta = displacement/(initial_ratio+Delta)
    err = 1
    while abs(err)>1e-8:
        delta_old = delta
        delta = (displacement + Delta*delta_transition/2*np.sqrt(np.pi/3)*math.erf(np.sqrt(3)*delta_old/delta_transition))/(initial_ratio + Delta)
        err = abs((delta-delta_old)/delta)
    return delta

def smoothing(curve,allowed_err):
    """_summary_

    Args:
        curve (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    N = len(curve[0])
    x = [curve[0][0]]
    y = [curve[1][0]]
    
    for i in range(1,N):
        if curve[0][i-1]<curve[0][i]:
            x.append(curve[0][i])
            y.append(curve[1][i])
            
    N = len(x)
    
    ################## tooth subdivision
    
    y_max = max(y)
    y_max_ind = y.index(y_max)
    
    i = 0
    while y[i]<y_max/2:
        i += 1
    
    angle = 0
    while abs(angle)<np.pi/6 and i<=y_max_ind:
        i += 1
        alpha = np.arctan2(y[i]-y[i-1],x[i]-x[i-1])
        beta = np.arctan2(y[i+1]-y[i],x[i+1]-x[i])
        angle = alpha-beta
    start_fillet = i
    
    left_flank = [x[:start_fillet],y[:start_fillet]]
    
    i = N-1
    while y[i]<y_max/2:
        i -= 1
    
    angle = 0
    while abs(angle)<np.pi/6 and i>=y_max_ind:
        i -= 1
        alpha = np.arctan2(y[i]-y[i+1],x[i]-x[i+1])
        beta = np.arctan2(y[i-1]-y[i],x[i-1]-x[i])
        angle = alpha-beta
    end_fillet = i
    
    right_flank = [x[end_fillet+1:],y[end_fillet+1:]]
    
    fillet_x = x[start_fillet:end_fillet+1]
    fillet_y = y[start_fillet:end_fillet+1]
    
    ################## fillet smoothing
    
    i = 0
    N_fillet = len(fillet_x)
    smooth_fillet_x  = [fillet_x[0]]
    smooth_fillet_y = [fillet_y[0]]
    # In the following cycle the point i+j is checked to be good or bad by taking the line between point i and i+j and checking if there are following points on the left of the line (left is taken sitting on i and looking at i+j). This should ensure that the line is a tangent line for the fillet.
    while i<N_fillet-1:
        j = 1
        pts_on_left = 1
        while i+j<N_fillet and pts_on_left != 0:
            pts_on_left = 0
            m = (fillet_y[i+j]-fillet_y[i])/(fillet_x[i+j]-fillet_x[i])
            x1 = fillet_x[i]
            y1 = fillet_y[i]
                
            if i+j<(N_fillet-1)-5:
                last_i = i+j+6
            else:
                last_i = N_fillet-1
            
            for k in range(i+j+1,last_i):
                check = fillet_y[k]-y1>(fillet_x[k]-x1)*m
                if check:
                    pts_on_left += 1
            j += 1
        i = i+j-1
        smooth_fillet_x.append(fillet_x[i])
        smooth_fillet_y.append(fillet_y[i])
       

    smooth_curve_x = np.append(np.append(left_flank[0],smooth_fillet_x),right_flank[0])
    smooth_curve_y = np.append(np.append(left_flank[1],smooth_fillet_y),right_flank[1])

    N = len(smooth_curve_x)
    
    ################## curve interpolation
    lengths = [0]
    for i in range(1,N):
        distance_i = np.linalg.norm([smooth_curve_x[i]-smooth_curve_x[i-1],smooth_curve_y[i]-smooth_curve_y[i-1]])
        lengths.append(lengths[i-1]+distance_i)
    
    # the following line creates a function that give the x for a given distance traveled on the curve
    length_LUT = interpolate.interp1d(lengths,smooth_curve_x)
    
    max_err = allowed_err+1
    number_of_points = 8
    points_l = np.linspace(lengths[0],lengths[-1],number_of_points,endpoint=True)
    # the first and last knot of a B-spline have to have multiplicity k+1 where k is the spline order
    points_l = np.append(np.full(3,points_l[0]),points_l)
    points_l = np.append(points_l,np.full(3,points_l[-1]))
    points_x = length_LUT(points_l)
    weights = np.ones(N)
    while max_err>allowed_err:
        
        interp = interpolate.make_lsq_spline(smooth_curve_x,smooth_curve_y,points_x,w=weights)
        
        interp_y = interp(smooth_curve_x)
        max_err = 0
        for i in range(N):
            new_err = abs(interp_y[i]-smooth_curve_y[i])
            if new_err>max_err:
                max_err = new_err
                idx_max_err = i
        weights[idx_max_err] += 1
        if idx_max_err > 0 | idx_max_err<N-1:
            weights[idx_max_err+1] += .5
            weights[idx_max_err-1] += .5
            
        x_to_add = smooth_curve_x[idx_max_err]
        if x_to_add not in points_x:
            idx_sort = np.searchsorted(points_x,x_to_add)
            points_x = np.insert(points_x,idx_sort,x_to_add)

    # removal of the multiplicity of the fist and last knot
    points_x = points_x[3:-3]
    points_y = interp(points_x)
    
    x_first = points_x[0]
    y_first = interp(x_first)
    while y_first>1e-10:
        m = 1/interpolate.splev(x_first,interp.tck,der=1,ext=0)
        x_first = x_first - y_first*m
        y_first = interp(x_first)
        
    x_last = points_x[-1]
    y_last = interp(x_last)
    while y_last>1e-10:
        m = 1/interpolate.splev(x_last,interp.tck,der=1,ext=0)
        x_last = x_last - y_last*m
        y_last = interp(x_last)
    
    points_x[0] = x_first
    points_x[-1] = x_last
    points_y[0] = 0
    points_y[-1] = 0
    
    return [points_x, points_y]

def rack_cut(cut_input):
    """_summary_

    Args:
        cut_input (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    rack, pinion, section_rotation, delta_pinion, rack_disp, axis_distance = cut_input
    
    for i in range(len(delta_pinion)):
        theta = delta_pinion[i]*np.pi/180 - section_rotation
        affinity_matrix = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta),rack_disp[i],-axis_distance]
        rack = rack.difference(affinity.affine_transform(pinion,affinity_matrix))
    
    return rack

if __name__ == '__main__':
#endregion
#region#################################################################################### Read configuration file    
    if len(sys.argv) < 2:
        cfg_file = 'Input\\defaultcfg2D.cfg'
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
    max_err = float(config['config']['max_err'])
    output_filename = config['config']['output_filename']
    pinion_filename = config['config']['pinion_filename']
    show_plots = bool(config['config']['show_plots'])

#endregion
#region#################################################################################### Read pinion input file     
    # Import of the pinion section from a .xyz file, each line of this file contain the coordinates of the points of the boundary of the section
    pinion_points = []
    with open(f'Input\\{pinion_filename}') as pinion_file:
        lines = pinion_file.readlines()
        # For each line I extract the x and y coordinates and assign them to the variable pinion_points
        for line in lines:
            x, y = line.strip().split(',')
            pinion_points.append([float(x),float(y)])
#endregion
#region#################################################################################### Constants definition       

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
#region#################################################################################### Arrays definition          
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
#region#################################################################################### Slicing                    
    # I create the two polygons, the pinion from the imported points and the rack to be cut is just a rectangle 
    pinion = Polygon(pinion_points)
    pinion = affinity.rotate(pinion,initial_rotation,origin=(0,0),use_radians=True)

    if show_plots:
        fig = plt.figure()
        pinion_ax = fig.add_subplot()
        pinion_ax.axis('equal')
        pinion_x, pinion_y = pinion.exterior.xy
        plt.plot(pinion_x,pinion_y,'.')
        plt.show()

    # Initialization of the two arrays output of the following for cycle
    slices = []

    # In this for cycles through the different slices of the rack, first the rack profile is computed using the boolean operations,
    # in order to address the helixicity of the teeth for each slice the pinion section is rotated accordingly, after the realization of the polygon
    # representing the slice of the rack, the points are extracted, the four points on the edges of the rack are eliminated.
    # Then, the array is ordered, the fillet on each tooth is smoothed out and the array is divided into curves.
    
    rack_blank = Polygon([[-l_left,0],[l_right,0],[l_right,rack_thickness],[-l_left,rack_thickness]])
    
    cut_input = []
    for height in plane_height:
        section_rotation = height/rp*np.sin(helix_angle)
        cut_input.append([rack_blank,pinion,section_rotation,delta_pinion,rack_disp,axis_distance])
    
    print("Cutting the slices...")
    num_proc = min([height_discretizations,cpu_count(),8])
    with Pool(num_proc) as pool:
        rack_polygon = list(tqdm.tqdm(pool.imap(rack_cut,cut_input),total=height_discretizations))
    
    for j, height in enumerate(plane_height):
        
        first_midpoint_x = variable_ratio_fun(delta_1_joint(-height/rp*np.tan(helix_angle)*180/np.pi,beta,alpha),rack_ratio,Delta_ratio,delta_trans)
        last_midpoint_x = variable_ratio_fun(delta_1_joint(ext_angle-height/rp*np.tan(helix_angle)*180/np.pi,beta,alpha),rack_ratio,Delta_ratio,delta_trans)
        
        
        rack_x, rack_y = np.array(rack_polygon[j].exterior.xy)
        
        # I find the index of the first corner in order to remove the unnecessary points
        ind = np.where(rack_x == -l_left)
        
        if rack_y[ind[0][0]] != 0:

            ind = ind[0][1]

        else:

            ind = ind[0][0]
        
        if rack_y[ind+1] != 0:

            rack_x = rack_x[::-1]
            rack_y = rack_y[::-1]

            ind = len(rack_x)-ind

        # I erase the 4 corners of the rack slice, keeping only the teeth
        rack_x = np.concatenate((rack_x[ind:],rack_x[:ind-4]))
        rack_y = np.concatenate((rack_y[ind:],rack_y[:ind-4]))
        

        indicies = np.where(rack_y==0)
    
        first_midpoint_index = (np.abs(rack_x-first_midpoint_x)).argmin()
        n = np.searchsorted(indicies[0],first_midpoint_index,'left')-1

        i = n
        slice = []
        while rack_x[indicies[0][i]]<last_midpoint_x:

            start = indicies[0][i]
            end = indicies[0][i+1]
            
            slice.append([rack_x[start:end+1],rack_y[start:end+1]])
            i +=2
        slices.append(slice)
#endregion
#region#################################################################################### Smoothing                  

    print("Smoothing the curves...")
    knots_slices = []
    num_proc = min([len(slices[0]),cpu_count(),8])
    with Pool(num_proc) as pool:
        for slice in tqdm.tqdm(slices):
            knots_slice = pool.map(smoothing,slice)
            knots_slices.append(knots_slice)
            
    #Use the following to develop the function smoothing
    # for slice in tqdm.tqdm(slices):
    #     knots_slice = []
    #     for curve in slice:
    #         knots_slice.append(smoothing(curve,max_err))
    #     knots_slices.append(knots_slice)

#endregion
#region#################################################################################### Plots                      
    # Inizialization of the figures used check the result and to debug the code

    if show_plots:
        fig = plt.figure()
        ax_3D = fig.add_subplot(projection='3d')
        ax_3D.axis('equal')
        fig = plt.figure()
        ax_2D = fig.add_subplot()
        ax_2D.axis('equal')
        fig = plt.figure()
        ax_fun = fig.add_subplot()        
        slice_to_check = 0 #np.ceil(height_discretizations/2)

        delta_wheel_ratio = [1/d for d in deriv(delta_wheel,rack_disp)]
        delta_pinion_ratio = [1/d for d in deriv(delta_pinion,rack_disp)]
        ax_fun.plot(rack_disp,delta_wheel_ratio)
        ax_fun.plot(rack_disp,delta_pinion_ratio)

        
        for i in range(height_discretizations):

            for j in range(len(knots_slices[i])):

                ax_3D.plot(knots_slices[i][j][0],knots_slices[i][j][1],plane_height[i],'b')

                if i == slice_to_check:

                    ax_2D.plot(slices[i][j][0],slices[i][j][1],'.r')
                    final_spline = interpolate.make_interp_spline(knots_slices[i][j][0],knots_slices[i][j][1],bc_type='natural')
                    final_spline_x = np.linspace(final_spline.t[0],final_spline.t[-1],1000,endpoint=True)
                    final_spline_y = final_spline(final_spline_x)
                    ax_2D.plot(final_spline_x,final_spline_y,'b')
                    ax_2D.plot(knots_slices[i][j][0],knots_slices[i][j][1],'og')

        ax_2D.grid(True)
        plt.show()
#endregion
#region#################################################################################### Spline/STEP file creation  

    result = cq.Workplane('XY')
    for k, slice in enumerate(knots_slices):

        for curve in slice:

            MSpline_vec = []
            spline_curve = []

            for i in range(len(curve[0])):

                MSpline_vec.append(cq.Vector(curve[0][i],curve[1][i],plane_height[k]))
            
            result.objects.append(cq.Edge.makeSpline(MSpline_vec))

    for j in range(len(knots_slices[0])):

        MSpline_vec_left = []
        MSpline_vec_right = []

        for k, slice in enumerate(knots_slices):

            MSpline_vec_left.append(cq.Vector(slice[j][0][0],slice[j][1][0],plane_height[k]))
            MSpline_vec_right.append(cq.Vector(slice[j][0][-1],slice[j][1][-1],plane_height[k]))

        result.objects.append(cq.Edge.makeSpline(MSpline_vec_left))
        result.objects.append(cq.Edge.makeSpline(MSpline_vec_right))

    cq.exporters.export(result,f'Output\\{date_str}{output_filename}.STEP')

#endregion