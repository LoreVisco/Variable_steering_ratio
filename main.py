# region ########### Header

import sys
import configparser as cp
from multiprocessing import Pool, cpu_count
import numpy as np
from matplotlib import pyplot as plt
from shapely import affinity
from shapely.geometry import Polygon
import math
import tqdm
from datetime import datetime
from scipy import interpolate
from steputils import p21

# endregion

# region ########### Functions definition


class extrapolation:
    def __init__(self, x0, interp):

        self.x0 = x0
        self.y0 = interp(x0)
        self.dy = interp(x0, 1)
        self.ddy = interp(x0, 2)
        self.dddy = interp(x0, 3)

    def __call__(self, x):

        y = self.y0+self.dy*(x-self.x0)+self.ddy * \
            (x-self.x0)**2+self.dddy*(x-self.x0)**3
        return y


class universal_joint:
    def __init__(self, alpha, beta):

        self.a = alpha*np.pi/180
        self.b = beta*np.pi/180

    def delta1(self, delta2):
        """
        INVERSE FUNCTION OF THE UNIVERSAL JOINT

        computes the angle of the input shaft delta_1 for a given configuration of the an universal joint, defined by the output shaft angle delta_2, the joint angle alpha and the phase angle beta. 



        In an universal joint two shafts are involved, input and output shafts, the angular position of the input shaft is called delta_1. Starting from this labeling of the shafts we can define the two angles alpha and beta.
        To do so let's create a fixed frame of reference on the end of the input shaft, the z axis aligned with the shaft axis, the x axis on the plane on which the two shafts lie in the direction of the output shaft and the y axis according with right hand rule.
        Then we create another fixed frame of reference rotated about the z-axis of an angle beta, that frame of reference is the one from which the angle delta_1 is measured, still according with the right-hand rule. 
        beta is called phase angle, in fact the function has a 180 degree periodicity and beta defines the starting point in the period, it is usually set to zero or 90 degree, values at which the transmission ratio (D delta_2)/(D delta_1) has a maximum and a minimum respectively.
        The function is built in such a way that delta_2=0 when delta_1=0, this is because we are often interested just in the difference between the input angle delta_1 and the output angle delta_2.
        the joint angle alpha is the acute angle between the input shaft and the output shaft, it should be positive definite.



        Args:
            delta_2 (float): output shaft angle in [deg]
            beta (float): phase angle in [deg] in the range (-90,90]
            alpha (float): joint angle in [deg] in the range [0,90)

        Returns:
            float: input shaft angle in [deg]
        """

        d = delta2*np.pi/180
        a = self.a
        b = self.b

        correction = np.pi * \
            np.ceil((d-np.pi/2+np.arctan2(np.tan(b), np.cos(a)))/np.pi)

        delta1 = np.arctan(
            np.cos(a)*np.tan(d+np.arctan2(np.tan(b), np.cos(a))))-b+correction

        return delta1*180/np.pi

    def delta2(self, delta1):
        """
        DIRECT FUNCTION OF THE UNIVERSAL JOINT

        computes the angle of the output shaft delta_2 for a given configuration of the an universal joint, defined by the input shaft angle delta_1, the joint angle alpha and the phase angle beta. 



        In an universal joint two shafts are involved, input and output shafts, the angular position of the input shaft is called delta_1. Starting from this labeling of the shafts we can define the two angles alpha and beta.
        To do so let's create a fixed frame of reference on the end of the input shaft, the z axis aligned with the shaft axis, the x axis on the plane on which the two shafts lie in the direction of the output shaft and the y axis according with right hand rule.
        Then we create another fixed frame of reference rotated about the z-axis of an angle beta, that frame of reference is the one from which the angle delta_1 is measured, still according with the right-hand rule. 
        beta is called phase angle, in fact the function has a 180 degree periodicity and beta defines the starting point in the period, it is usually set to zero or 90 degree, values at which the transmission ratio (D delta_2)/(D delta_1) has a maximum and a minimum respectively.
        The function is built in such a way that delta_2=0 when delta_1=0, this is because we are often interested just in the difference between the input angle delta_1 and the output angle delta_2.
        the joint angle alpha is the acute angle between the input shaft and the output shaft, it should be positive definite.



        Args:
            delta_1 (float): input shaft angle in [deg]
            beta (float): phase angle in [deg] in the range (-90,90]
            alpha (float): joint angle in [deg] in the range [0,90)

        Returns:
            float: output shaft angle in [deg]
        """
        a = self.a
        b = self.b
        d = delta1*np.pi/180

        correction = np.pi*np.ceil((d-np.pi/2+b)/np.pi)

        delta2 = np.arctan2(np.tan(d+b), np.cos(a)) - \
            np.arctan2(np.tan(b), np.cos(a))+correction

        return delta2*180/np.pi


class rack_variable_ratio:

    def __init__(self, initial_ratio, Delta_ratio, delta_transition) -> None:

        self.initial_ratio = initial_ratio
        self.Delta_ratio = Delta_ratio
        self.delta_transition = delta_transition

    def disp(self, steering):
        """
        DIRECT FUNCTION OF RACK DISPLACEMENT FOR A VARIABLE STEERING RATIO

        This function computes the rack displacement for a given steering angle and the chosen steering ratio profile, which is given by the three parameters: initial_ratio, Delta_ratio, delta_transition.



        The variable steering ratio function is chosen to be of the kind 1-e^(delta)^2, this allows to have a very smoothly changing ratio with a zero derivative for delta=0. In order to compute the position, the intergral function have been found. The function computes the displacement of the rack as a function of steering angle for the given profile, where the initial ratio is the desired steering ratio for delta=0, Delta_ratio is the difference between final and initial steering ratio and delta_transition is the steering angle at which the final steering ratio have to be reached.



        Args:
            delta (float): input steering angle in [deg]
            initial_ratio (float): steering ratio in [mm/deg] at delta=0
            Delta_ratio (float): difference between final and initial steering ratio (negative values are allowed)
            delta_transition (float): steering angle in [deg] at which the final steering is reached

        Returns:
            float: steering rack displacemente
        """

        tau_i = self.initial_ratio
        D_tau = self.Delta_ratio
        d_t = self.delta_transition
        d = steering

        disp = (tau_i+D_tau)*d - D_tau*d_t/2 * \
            np.sqrt(np.pi/3)*math.erf(d/d_t*np.sqrt(3))
        return disp

    def steer(self, displacement):
        """
        INVERSE FUNCTION OF RACK DISPLACEMENT FOR A VARIABLE STEERING RATIO

        This function computes the steering angle for a given rack displacement and the chosen steering ratio profile, which is given by the three parameters: initial_ratio, Delta_ratio, delta_transition.



        The variable steering ratio function is chosen to be of the kind 1-e^(delta)^2, this allows to have a very smoothly changing ratio with a zero derivative for delta=0. In order to compute the steering angle, the intergral function have been found and then numerically inverted. The function computes the steering angle as a function of rack displacement for the given profile, where the initial ratio is the desired steering ratio for delta=0, Delta_ratio is the difference between final and initial steering ratio and delta_transition is the steering angle at which the final steering ratio have to be reached.



        Args:
            displacement (float): steering rack displacement in [mm]
            initial_ratio (float): steering ratio in [mm/deg] at delta=0
            Delta_ratio (float): difference between final and initial steering ratio (negative values are allowed)
            delta_transition (float): steering angle in [deg] at which the final steering is reached


        Returns:
            float: steering angle in [deg]
        """

        tau_i = self.initial_ratio
        D_tau = self.Delta_ratio
        d_t = self.delta_transition
        disp = displacement

        d = disp/(tau_i+D_tau)  # initial guess
        err = 1
        while abs(err) > 1e-10:
            d_old = d
            d = (disp + D_tau*d_t/2*np.sqrt(np.pi/3) *
                 math.erf(np.sqrt(3)*d_old/d_t))/(tau_i + Delta_ratio)
            err = abs((d-d_old)/d)
        return d


def deriv(y, x):
    """
    DERIVATIVE FUNCTION OF Y WITH RESPECT TO X

    It computes the numeric derivative of y with respect to x.
    The function takes two array x and y and performs the numerical derivative using the forward difference, the last value is linearly extrapolated from the previous values. It doesn't need x and y to be of the same length, the length of the output will be equal to the length of the shortest.



    Args:
        y (float): array that defines the function of which to compute the derivative
        x (float): array that defines the variable against which to compute the derivative

    Returns:
        float: array of the same length of the shortest between x and y representing dy/dx
    """
    length = min([len(y), len(x)])
    dy = np.empty(length)
    for i in range(length-1):
        dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    dy[i+1] = dy[i]+(x[i+1]-x[i])*(dy[i]-dy[i-1])/(x[i]-x[i-1])
    return dy


def smoothing(input):
    """
    This function is prepared for parallel computing. It takes as input an array with x and y coordinates of the tooth curve to be smoothen out and, first finds the fillet, delete the unwanted points, and then it creates an interpolating spline that approximate the input curve with the maximum error defined in the input. It gives the set of points of the interpolating spline.



    Args:
        input (float): [curve, max_allowed_error] the variable "curve" is an array of floats with x and y coordinates of each point: curve = [[x],[y]], x and y have to be of the same shape. max_allowed_error is the maximum error in y direction between the computed spline and the original points of the input curve in [mm].

    Returns:
        float: An array of points of the kind [[x],[y]] with the coorinates of the points needed to create the interpolating spline
    """
    curve, allowed_err = input
    N = len(curve[0])
    x = []
    y = []

    i = 0
    while i < N-1:
        x.append(curve[0][i])
        y.append(curve[1][i])
        j = 1
        while curve[0][i+j] <= curve[0][i] or curve[0][-1] <= curve[0][i+j]:
            j += 1
            if i+j == N:
                break
        i += j
    x.append(curve[0][-1])
    y.append(curve[1][-1])
    N = len(x)

    # tooth subdivision

    y_max = max(y)
    y_max_ind = y.index(y_max)

    i = 0
    while y[i] < y_max/2:
        i += 1

    angle = 0
    while abs(angle) < np.pi/6 and i <= y_max_ind:
        i += 1
        alpha = np.arctan2(y[i]-y[i-1], x[i]-x[i-1])
        beta = np.arctan2(y[i+1]-y[i], x[i+1]-x[i])
        angle = alpha-beta
    start_fillet = i

    left_flank = [x[:start_fillet], y[:start_fillet]]

    i = N-1
    while y[i] < y_max/2:
        i -= 1

    angle = 0
    while abs(angle) < np.pi/6 and i >= y_max_ind:
        i -= 1
        alpha = np.arctan2(y[i]-y[i+1], x[i]-x[i+1])
        beta = np.arctan2(y[i-1]-y[i], x[i-1]-x[i])
        angle = alpha-beta
    end_fillet = i

    right_flank = [x[end_fillet+1:], y[end_fillet+1:]]

    fillet_x = x[start_fillet:end_fillet+1]
    fillet_y = y[start_fillet:end_fillet+1]

    # fillet smoothing

    i = 0
    N_fillet = len(fillet_x)
    smooth_fillet_x = [fillet_x[0]]
    smooth_fillet_y = [fillet_y[0]]
    # In the following cycle the point i+j is checked to be good or bad by taking the line between point i and i+j and checking if there are following points on the left of the line (left is taken sitting on i and looking at i+j). This should ensure that the line is a tangent line for the fillet.
    while i < N_fillet-2:
        j = 0
        pts_on_left = 1
        while i+j < N_fillet-1 and pts_on_left != 0:
            j += 1
            pts_on_left = 0
            m = (fillet_y[i+j]-fillet_y[i])/(fillet_x[i+j]-fillet_x[i])
            x1 = fillet_x[i]
            y1 = fillet_y[i]

            if i+j < (N_fillet-1)-5:
                last_i = i+j+6
            else:
                last_i = N_fillet-1

            for k in range(i+j+1, last_i):
                check = fillet_y[k]-y1 > (fillet_x[k]-x1)*m
                if check:
                    pts_on_left += 1
        i += j
        if i != N_fillet-2:
            smooth_fillet_x.append(fillet_x[i])
            smooth_fillet_y.append(fillet_y[i])

    if N_fillet > 1:
        smooth_fillet_x.append(fillet_x[-1])
        smooth_fillet_y.append(fillet_y[-1])

    smooth_curve_x = np.append(
        np.append(left_flank[0], smooth_fillet_x), right_flank[0])
    smooth_curve_y = np.append(
        np.append(left_flank[1], smooth_fillet_y), right_flank[1])

    N = len(smooth_curve_x)

    # curve interpolation
    lengths = [0]
    for i in range(1, N):
        distance_i = np.linalg.norm(
            [smooth_curve_x[i]-smooth_curve_x[i-1], smooth_curve_y[i]-smooth_curve_y[i-1]])
        lengths.append(lengths[i-1]+distance_i)

    # the following line creates a function that give the x for a given distance traveled on the curve
    length_LUT = interpolate.interp1d(lengths, smooth_curve_x)

    max_err = allowed_err+1
    number_of_points = 2
    points_l = np.linspace(
        lengths[0], lengths[-1], number_of_points, endpoint=True)
    # the first and last knot of a B-spline have to have multiplicity k+1 where k is the spline order
    points_x = []
    points_l = np.append(np.full(3, points_l[0]), points_l)
    points_l = np.append(points_l, np.full(3, points_l[-1]))
    points_x = length_LUT(points_l)
    weights = np.ones(N)
    while max_err > allowed_err:

        interp = interpolate.make_lsq_spline(
            smooth_curve_x, smooth_curve_y, points_x, w=weights)

        interp_y = interp(smooth_curve_x)
        max_err = 0
        for i in range(N):
            new_err = abs(interp_y[i]-smooth_curve_y[i])
            if new_err > max_err:
                max_err = new_err
                idx_max_err = i
        weights[idx_max_err] += 1
        if idx_max_err > 0 | idx_max_err < N-1:
            weights[idx_max_err+1] += .5
            weights[idx_max_err-1] += .5

        x_to_add = smooth_curve_x[idx_max_err]
        if x_to_add not in points_x:
            idx_sort = np.searchsorted(points_x, x_to_add)
            points_x = np.insert(points_x, idx_sort, x_to_add)

    # removal of the multiplicity of the fist and last knot
    points_x = points_x[3:-3]
    points_y = interp(points_x)

    x0 = points_x[0]
    extr = extrapolation(x0, interp)

    if abs(interp(x0)) > 1e-8:

        a = b = x0

        while np.sign(extr(a)) == 1:
            a -= allowed_err

        while np.sign(extr(b)) == -1:
            b += allowed_err

        c = (a+b)/2

        while abs(extr(c)) > 1e-8:

            if np.sign(extr(a)*extr(c)) == -1:
                b = c
            else:
                a = c
            c = (a+b)/2
        points_x[0] = c
    points_y[0] = 0

    x0 = points_x[-1]
    extr = extrapolation(x0, interp)
    if abs(interp(x0)) > 1e-8:
        a = b = x0

        while np.sign(extr(a)) == -1:
            a -= allowed_err

        while np.sign(extr(b)) == 1:
            b += allowed_err

        c = (a+b)/2

        while abs(extr(c)) > 1e-8:
            if np.sign(extr(a)*extr(c)) == -1:
                b = c
            else:
                a = c
            c = (a+b)/2
        points_x[-1] = c
    points_y[-1] = 0

    final_spline = interpolate.make_interp_spline(
        points_x, points_y, bc_type='natural')
    control_points_y = final_spline.c
    control_points_x = [sum(final_spline.t[list(range(i+1, i+4))]) /
                        3 for i in range(len(final_spline.t)-4)]
    knots = final_spline.t

    return [points_x, points_y, control_points_x, control_points_y, knots]


def rack_cut(input):
    """
    This function is prepared for parallel computing. It creates a polygon representing a single slice of the steering rack by means of a series of boolean operations using the pinion cutter polygon.

    Args:
        input (any): it is an array of the kind: [rack, pinion, section_rotation, delta_pinion, rack_disp, axis_distance] where:
            rack: is the blank polygon of the rack to be cut
            pinion: is the polygon representing se section of the cutter pinion
            section_rotation: the angle in [deg] of fixed rotation to address the helicity of the pinion
            delta_pinion: an array of all the angular position of the pinion in [deg]
            rack_disp: an array of all the displacements in [mm] of the rack for each angular position of the pinion
            axis_distance: the distance between the axis of the pinion and the rack surface in [mm]

    Returns:
        polygon: polygon representing the single slice of the steering rack
    """

    rack, pinion, section_rotation, delta_pinion, rack_disp, axis_distance = input

    # fig = plt.figure()
    # ax_boolean = fig.add_subplot()
    # ax_boolean.axis('equal')
    # ax_boolean.set_ylabel('y [mm]')
    # ax_boolean.set_xlabel('x [mm]')
    # ax_boolean.set_title('Boolan subtraction')
    # print_index = round(len(delta_pinion)/3)-5

    for i in range(len(delta_pinion)):
        theta = delta_pinion[i]*np.pi/180 - section_rotation
        affinity_matrix = [np.cos(
            theta), -np.sin(theta), np.sin(theta), np.cos(theta), rack_disp[i], -axis_distance]
        rack = rack.difference(
            affinity.affine_transform(pinion, affinity_matrix))

        # only to generate picture for thesis
    #     if i>print_index-15 and i<=print_index:

    #         if i == print_index:
    #             rackbool_x, rackbool_y = rack.exterior.xy
    #             ax_boolean.fill(rackbool_x, rackbool_y,facecolor='#1f77b4', alpha=0.5, label='Rack')
    #         if i%3==0:
    #             pinionbool_x, pinionbool_y = affinity.affine_transform(pinion, affinity_matrix).exterior.xy
    #             ax_boolean.fill(pinionbool_x, pinionbool_y,facecolor='#ff7f0e',  alpha=0.2)

    # ax_boolean.set_xlim(-15, 20)
    # plt.savefig('Output/boolean_subtraction.pdf',
    #                 dpi='figure', format='pdf')

    return rack


def write_step_obj(slices, fname):
    """_summary_

    Args:
        knots_slice (_type_): _description_
        fname (_type_): _description_
    """
    step_obj = p21.new_step_file()

    step_obj.header.set_file_description(('',), '2;1')
    step_obj.header.set_file_name(name=fname, time_stamp=p21.timestamp(
    ), organization=('LV', 'Visconti_L'), autorization='LV')
    step_obj.header.set_file_schema(
        ('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }', ))

    data = step_obj.new_data_section()

    data.add(p21.simple_instance(
        '#1', 'APPLICATION_CONTEXT', ('automotive_design', )))
    data.add(p21.simple_instance('#2', 'APPLICATION_PROTOCOL_DEFINITION',
             ('draft international standard', 'automotive_design', 1998, p21.reference('#1'))))
    data.add(p21.simple_instance('#3', 'PRODUCT',
             ('rack', '', '', p21.reference('#4'))))
    data.add(p21.simple_instance('#4', 'PRODUCT_CONTEXT',
             ('NONE', p21.reference('#1'), 'mechanical')))
    data.add(p21.simple_instance('#5', 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE',
             ('EVERYTHING', '', p21.reference('#3'), p21.enum('.NOT_KNOWN.'))))
    data.add(p21.simple_instance(
        '#6', 'PRODUCT_RELATED_PRODUCT_CATEGORY', ('part', '', p21.reference('#3'))))
    data.add(p21.simple_instance('#7', 'PRODUCT_DEFINITION_CONTEXT',
             ('part definition', p21.reference('#1'), 'design')))
    data.add(p21.simple_instance('#8', 'PRODUCT_DEFINITION',
             ('NOT KNOWN', '', p21.reference('#5'), p21.reference('#7'))))
    data.add(p21.simple_instance('#9', 'PRODUCT_DEFINITION_SHAPE',
             ('NONE', 'NONE', p21.reference('#8'))))
    data.add(p21.simple_instance('#10', 'SHAPE_DEFINITION_REPRESENTATION',
             (p21.reference('#9'), p21.reference('#11'))))
    data.add(p21.simple_instance('#11', 'GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION',
             ('curves', (p21.reference('#17'), ), p21.reference('#12'))))

    entitylist = []
    entitylist.append(p21.entity('GEOMETRIC_REPRESENTATION_CONTEXT', (3, )))
    entitylist.append(p21.entity(
        'GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT', (p21.reference('#13', ), )))
    entitylist.append(p21.entity('GLOBAL_UNIT_ASSIGNED_CONTEXT', ((
        p21.reference('#14'), p21.reference('#15'), p21.reference('#16')), )))
    entitylist.append(p21.entity(
        'REPRESENTATION_CONTEXT', ('NONE', 'WORKSPACE')))
    data.add(p21.complex_entity_instance('#12', entitylist))

    data.add(p21.simple_instance('#13', 'UNCERTAINTY_MEASURE_WITH_UNIT',
             (1E-05, p21.reference('#14'), 'distance_accuracy_value', 'NONE')))

    entitylist = []
    entitylist.append(p21.entity('LENGTH_UNIT', ()))
    entitylist.append(p21.entity('NAMED_UNIT', (p21.unset_parameter('*'), )))
    entitylist.append(p21.entity(
        'SI_UNIT', (p21.enum('.MILLI.'), p21.enum('.METRE.'))))
    data.add(p21.complex_entity_instance('#14', entitylist))

    entitylist = []
    entitylist.append(p21.entity('NAMED_UNIT', (p21.unset_parameter('*'), )))
    entitylist.append(p21.entity('PLANE_ANGLE_UNIT', ()))
    entitylist.append(p21.entity(
        'SI_UNIT', (p21.unset_parameter('$'), p21.enum('.RADIAN.'))))
    data.add(p21.complex_entity_instance('#15', entitylist))

    entitylist = []
    entitylist.append(p21.entity('NAMED_UNIT', (p21.unset_parameter('*'), )))
    entitylist.append(p21.entity(
        'SI_UNIT', (p21.unset_parameter('$'), p21.enum('.STERADIAN.'))))
    entitylist.append(p21.entity('SOLID_ANGLE_UNIT', ()))
    data.add(p21.complex_entity_instance('#16', entitylist))

    data.add(p21.simple_instance(
        '#17', 'DRAUGHTING_PRE_DEFINED_COLOUR', ('black', )))
    data.add(p21.simple_instance(
        '#18', 'DRAUGHTING_PRE_DEFINED_CURVE_FONT', ('continuous', )))
    data.add(p21.simple_instance('#19', 'POSITIVE_LENGTH_MEASURE', (0.02, )))
    data.add(p21.simple_instance('#20', 'CURVE_STYLE', ('', p21.reference(
        '#18'), p21.reference('#19'), p21.reference('#17'))))
    data.add(p21.simple_instance(
        '#21', 'PRESENTATION_STYLE_ASSIGNMENT', p21.reference('#20', )))

    curves_ref = []
    curves_ref_style = []

    for i, slice in enumerate(slices):
        curves_in_slice_ref = []
        curves_in_slice_ref_style = []

        for j, curve in enumerate(slice):

            curveID = f'{str(i+1)}{str(j+1).zfill(int(math.log10(len(slice)))+2)}'
            curves_in_slice_ref.append(p21.reference(f'#{curveID}'))
            curves_in_slice_ref_style.append(p21.reference(f'#3000{curveID}'))
            points_ref = []
            for k in range(len(curve[2])):

                pointID = f'{curveID}{str(k+1).zfill(int(math.log10(len(curve[2])))+1)}'
                point_params = (f's:{str(i+1)}_c:{str(j+1)}_p:{str(k+1)}',
                                (curve[2][k], curve[3][k], plane_height[i]))
                data.add(p21.simple_instance(
                    f'#{pointID}', 'CARTESIAN_POINT', point_params))
                points_ref.append(p21.reference(f'#{pointID}'))

            knots = list(curve[4][3:-3])
            mult = [4]+[1]*(len(knots)-2)+[4]
            curve_params = (f's:{str(i+1)}_c:{str(j+1)}', 3, points_ref, p21.enum('.UNSPECIFIED.'),
                            p21.enum('.F.'), p21.enum('.F.'), mult, knots, p21.enum('.UNSPECIFIED.'))
            data.add(p21.simple_instance(
                f'#1000{curveID}', 'B_SPLINE_CURVE_WITH_KNOTS', curve_params))
            data.add(p21.simple_instance(f'#2000{curveID}', 'COMPOSITE_CURVE_SEGMENT', (p21.enum(
                '.CONTINUOUS.'), p21.enum('.T.'), p21.reference(f'#1000{curveID}'))))
            data.add(p21.simple_instance(f'#3000{curveID}', 'STYLED_ITEM', ('', (p21.reference(
                '#21'), ), p21.reference(f'#{curveID}'))))
            data.add(p21.simple_instance(f'#{curveID}', 'COMPOSITE_CURVE', (
                f's:{str(i+1)}_c:{str(j+1)}', (p21.reference(f'#2000{curveID}'), ), p21.enum('.F.'))))

        curves_ref.extend(curves_in_slice_ref)
        curves_ref_style.extend(curves_in_slice_ref_style)

    data.add(p21.simple_instance(
        '#22', 'GEOMETRIC_SET', ('list of curves', curves_ref)))
    data.add(p21.simple_instance('#23', 'GMECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION',
             ('list of curves with style', curves_ref_style)))

    return step_obj
# endregion


if __name__ == '__main__':

    # region ########### Read configuration file

    if len(sys.argv) < 2:
        cfg_file = 'Input/defaultcfg2D.cfg'
    else:
        cfg_file = sys.argv[1]

    config = cp.ConfigParser()
    config.read(cfg_file)

    # see configuration file for contants description
    mn = float(config['config']['mn'])
    rp = float(config['config']['rp'])
    helix_angle = float(config['config']['helix_angle'])
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
    show_plots = config['config']['show_plots'].lower() in (
        'true', 't', 't.', 'vero', 'v', 'v.', 'yes', 'y', 'y.', 'si', 's', '1')

# endregion

# region ########### Read pinion input file

    # Import of the pinion section from a .xyz file, each line of this file contain the coordinates of the points of the boundary of the section
    pinion_points = []
    with open(f'Input/{pinion_filename}') as pinion_file:
        lines = pinion_file.readlines()
        # For each line I extract the x and y coordinates and assign them to the variable pinion_points
        for line in lines:
            x, y = line.strip().split(',')
            pinion_points.append([float(x), float(y)])

# endregion

# region ########### Constants definition

    date_str = datetime.now().strftime("%y%m%d_%H%M_")

    find_tip_radius = [np.linalg.norm(pts) for pts in pinion_points]
    tip_radius = max(find_tip_radius)
    tip_index = find_tip_radius.index(tip_radius)
    initial_rotation = np.pi/2 - \
        np.arctan2(pinion_points[tip_index][1], pinion_points[tip_index][0])
    Ujoint = universal_joint(alpha, beta)
    var_ratio = rack_variable_ratio(rack_ratio, Delta_ratio, delta_trans)
    m = mn/np.cos(helix_angle*np.pi/180)
    extreme_steering_wheel_angle = var_ratio.steer(rack_stroke)
    z = round(2*rp/m)
    angle_bw_teeth = 360/z
    N_complete_teeth = math.ceil(Ujoint.delta2(
        extreme_steering_wheel_angle)/angle_bw_teeth + 1)
    ext_angle = N_complete_teeth*angle_bw_teeth
    axis_distance = rp-m*(1-c)
    delta_pinion_tip_exit = np.arccos(axis_distance/tip_radius)*180/np.pi
    delta_pinion_tooth_completion = rack_height / \
        2/rp*np.tan(helix_angle*np.pi/180)*180/np.pi
    delta_pinion_overtravel = delta_pinion_tip_exit + delta_pinion_tooth_completion
    l_right = var_ratio.disp(ext_angle + delta_pinion_overtravel) + tip_radius
    l_left = var_ratio.disp(delta_pinion_overtravel) + tip_radius
    rack_thickness = tip_radius - axis_distance + 1

    slice_to_check = int(np.floor(height_discretizations/2))  # 5

# endregion

# region ########### Arrays definition

    delta_pinion = np.arange(-delta_pinion_overtravel,
                             ext_angle+delta_pinion_overtravel,
                             deg_step)

    N = len(delta_pinion)

    delta_wheel = [Ujoint.delta1(d) for d in delta_pinion]
    # delta_pinion = delta_wheel # Activate this line if you want to deactivate cardanic joint correction

    rack_disp = [var_ratio.disp(d) for d in delta_wheel]

    plane_height = np.linspace(-rack_height/2, rack_height/2,
                               height_discretizations, endpoint=True)

    if show_plots:
        fig = plt.figure()
        ax_fun = fig.add_subplot()

        delta_wheel_ratio = [d for d in deriv(rack_disp, delta_wheel)]
        delta_pinion_ratio = [d for d in deriv(rack_disp, delta_pinion)]
        ax_fun.plot(delta_wheel, delta_wheel_ratio, label='steering-rack')
        ax_fun.plot(delta_wheel, delta_pinion_ratio, label='pinion-rack')
        ax_fun.set_title('Computed transmission ratio laws')
        ax_fun.set_xlim(0, extreme_steering_wheel_angle)
        ax_fun.set_ylabel('transmission ratio [mm/deg]')
        ax_fun.set_xlabel('steering wheel position [deg]')
        ax_fun.legend()
        ax_fun.grid(True)
        plt.savefig('Output/transmission_ratios.pdf',
                    dpi='figure', format='pdf')


# endregion

# region ########### Slicing

    pinion = Polygon(pinion_points)
    pinion = affinity.rotate(pinion, initial_rotation,
                             origin=(0, 0), use_radians=True)

    # tips = 0
    # radius = 0
    # local_radius = [np.linalg.norm(pts) for pts in pinion_points]
    # while tips != z:
    #     radius += (tip_radius-radius)/2
    #     tips = sum([loc_rad>radius for loc_rad in local_radius])

    # tips_idx = np.where([loc_rad>radius for loc_rad in local_radius])

    # tips_xy = [[pinion_points[idx][0],pinion_points[idx][1]] for idx in tips_idx]

    if show_plots:
        fig = plt.figure()
        pinion_ax = fig.add_subplot()
        pinion_ax.axis('equal')
        pinion_x, pinion_y = pinion.exterior.xy
        plt.fill(pinion_x, pinion_y, alpha=0.5)
        plt.show()

    slices = []

    rack_blank = Polygon([[-l_left, 0], [l_right, 0],
                         [l_right, rack_thickness], [-l_left, rack_thickness]])

    cut_input = []
    for height in plane_height:
        section_rotation = height/rp*np.sin(helix_angle)
        cut_input.append([rack_blank, pinion, section_rotation,
                         delta_pinion, rack_disp, axis_distance])

    print("Cutting the slices...")
    num_proc = min([height_discretizations, cpu_count(), 8])
    with Pool(num_proc) as pool:
        rack_polygon = list(
            tqdm.tqdm(pool.imap(rack_cut, cut_input), total=height_discretizations))
        
    

    if show_plots:
        fig = plt.figure()
        ax_3D = fig.add_subplot(projection='3d')
        ax_3D.axis('equal')
        for k, slice in enumerate(rack_polygon):
            x, y = slice.exterior.xy
            ax_3D.plot(x, y, plane_height[k])

        fig = plt.figure()
        ax_centrodes = fig.add_subplot()
        ax_centrodes.set_title('Centrodes')
        inverse_ratio = [1/tau for tau in delta_pinion_ratio]
        cent_x = [1/tau*np.sin(delta_pinion[i]*np.pi/180)
                  for i, tau in enumerate(delta_pinion_ratio)]
        cent_y = [1/tau*np.cos(delta_pinion[i]*np.pi/180)
                  for i, tau in enumerate(delta_pinion_ratio)]
        rack_pol_x, rack_pol_y = rack_polygon[slice_to_check].exterior.xy
        rack_pol_y = [y+axis_distance for y in rack_pol_y]
        ax_centrodes.fill(rack_pol_x, rack_pol_y, alpha=0.5)
        ax_centrodes.fill(pinion_x, pinion_y, alpha=0.5)
        ax_centrodes.plot(rack_disp, inverse_ratio, label='rack centrode')
        pinion_centrode = ax_centrodes.plot(
            cent_x, cent_y, label='pinion centrode')
        ax_centrodes.plot([-x for x in cent_x], cent_y, color='#ff7f0e')
        ax_centrodes.legend()
        ax_centrodes.axis('equal')
        ax_centrodes.set_ylabel('y [mm]')
        ax_centrodes.set_xlabel('x [mm]')
        ax_centrodes.set_xlim(-10, 40)
        ax_centrodes.set_ylim(-10, 15)
        plt.savefig('Output/centrodes.pdf',
                    dpi='figure', format='pdf')
        plt.show()

    for j, height in enumerate(plane_height):

        pinion_helix_offset = height/rp*np.tan(helix_angle)*180/np.pi
        first_midpoint_x = var_ratio.disp(Ujoint.delta1(-pinion_helix_offset))
        last_midpoint_x = var_ratio.disp(Ujoint.delta1(
            ext_angle+angle_bw_teeth-pinion_helix_offset))

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
        rack_x = np.concatenate((rack_x[ind:], rack_x[:ind-4]))
        rack_y = np.concatenate((rack_y[ind:], rack_y[:ind-4]))

        indicies = np.where(rack_y == 0)

        first_midpoint_index = (np.abs(rack_x-first_midpoint_x)).argmin()
        n = np.searchsorted(indicies[0], first_midpoint_index, 'left')-1

        i = n
        slice = []
        while rack_x[indicies[0][i+1]] < last_midpoint_x:

            start = indicies[0][i]
            end = indicies[0][i+1]

            slice.append([rack_x[start:end+1], rack_y[start:end+1]])
            i += 2
        slices.append(slice)

    for slice in slices:
        if len(slice) != N_complete_teeth+1:
            raise ValueError('Pointing!')

    if show_plots:
        fig = plt.figure()
        ax_2D = fig.add_subplot()
        ax_2D.axis('equal')
        for curve in slices[slice_to_check]:
            ax_2D.plot(curve[0], curve[1], '.r')


# endregion

# region ########### Smoothing

    print("Smoothing the curves...")
    knots_slices = []
    num_proc = min([len(slices[0]), cpu_count(), 8])
    with Pool(num_proc) as pool:
        for slice in tqdm.tqdm(slices):
            input = []
            for curve in slice:
                input.append([curve, max_err])
            knots_slice = pool.map(smoothing, input)
            knots_slices.append(knots_slice)

    # Use the following to develop the smoothing function
    # for slice in tqdm.tqdm(slices):
    #     knots_slice = []
    #     for curve in slice:
    #         knots_slice.append(smoothing([curve,max_err]))
    #     knots_slices.append(knots_slice)

# endregion

# region ########### Plots
    # Inizialization of the figures used check the result and to debug the code

    if show_plots:
        fig = plt.figure()
        ax_3D = fig.add_subplot(projection='3d')
        ax_3D.axis('equal')

        for i in range(height_discretizations):

            for j in range(len(knots_slices[i])):

                final_spline = interpolate.make_interp_spline(
                    knots_slices[i][j][0], knots_slices[i][j][1], bc_type='natural')
                final_spline_x = np.linspace(
                    final_spline.t[0], final_spline.t[-1], 1000, endpoint=True)
                final_spline_y = final_spline(final_spline_x)
                ax_3D.plot(final_spline_x, final_spline_y,
                           plane_height[i], 'b')

                if i == slice_to_check:

                    final_spline_y_comp = interpolate.splev(
                        final_spline_x, (knots_slices[i][j][4], knots_slices[i][j][3], 3))
                    ax_2D.plot(final_spline_x, final_spline_y, 'b')
                    ax_2D.plot(knots_slices[i][j][0],
                               knots_slices[i][j][1], 'og')
                    ax_2D.plot(knots_slices[i][j][2],
                               knots_slices[i][j][3], '--om')

        ax_2D.grid(True)
        plt.show()

# endregion

# region ########### Spline/STEP file creation

    fname = f'{date_str}{output_filename}.STEP'

    stepfile = write_step_obj(knots_slices, fname)

    stepfile.save(f'Output/{fname}')

# endregion
