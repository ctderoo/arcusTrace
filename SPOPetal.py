from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.special import wofz,beta,gamma
import time
import os
import pdb
import pickle

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic
import PyXFocus.transformMod as transM
import PyXFocus.analyses as anal

import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusRays as ArcRays

####################################################################
# SPO-Related Functions
####################################################################

def id_XOU_for_rays(ray_object,xou_dict):
    rays = ray_object.yield_prays()
    xou_hit = empty(len(rays[0]))
    xou_hit.fill(NaN)
    
    for key in xou_dict.keys():
        clocked_rays = ArcUtil.do_ray_transform_to_coordinate_system(rays,xou_dict[key].xou_coords)
        surf.flat(clocked_rays)
        radii = sqrt(clocked_rays[1]**2 + clocked_rays[2]**2)
        bool_list = logical_and(logical_and(logical_and(ray_object.weight > 0, \
            radii > xou_dict[key].inner_radius),radii < xou_dict[key].outer_radius), \
            abs(clocked_rays[1]) <= xou_dict[key].azwidth/2)
        xou_hit[bool_list] = xou_dict[key].xou_num
    ray_object.xou_hit = xou_hit

def XOUTrace(ray_object,xou):
    """
    Trace a set of rays through an XOU. Incorporates ray plate selection and
    plate-by-plate weighting, but does not yet incorporate pore width effects.
    #Inputs:
    # ray_object -- a ray class from arcusRays
    # xou -- an xou object from ArcusComponents.
    #Outputs:
    # xou_rays = a ray object centered on the SPO, with:
    #    x - the radial coordinates, zeroed at the center of the SPO radius,
    #    y - the azimuthal coordinates, zeroed at the center of the SPO module
    #    z - the axial coordinate, with positive z pointed towards the source,
    #    and zeroed at the center of the SPO interface.
    """

    # Generating an internal ray object to play nicely with PyXFocus routines.
    input_rays = ray_object.yield_prays()
    wavelength = ray_object.wave
    weight = ray_object.weight

    int_rays = ArcUtil.do_ray_transform_to_coordinate_system(input_rays,xou.xou_coords)
    surf.flat(int_rays)
    ray_radius = sqrt(int_rays[1]**2+int_rays[2]**2)
    
    if xou.pore_vignette == True:
        weight *= ArcPerf.apply_support_weighting(int_rays,support = 'PoreStructure')
    
    for r in xou.plate_radii:
        # Collect relevant rays: tr_ind are the rays that should be traced,
        # v_ind are the rays that should be vignetted (impacting plates).
        # Note that the sidewall vignetting is completely ignored.
        tr_ind = logical_and(ray_radius > r,ray_radius < r + xou.pore_space)
        v_ind = logical_and(ray_radius > r + xou.pore_space, ray_radius < r + xou.plate_height)
        weight[v_ind] = 0
        # In case our sampling is so sparse that there are no rays fulfilling this condition.
        if sum(tr_ind) == 0: 
            continue
        
        # Changed spoPrimary/spoSecondary to functionally be f - DeltaX, as it is in Eq. 14 in Willingale et al. 2013 (https://arxiv.org/ftp/arxiv/papers/1307/1307.1709.pdf)
        surf.spoPrimary(int_rays,r,xou.z0,ind = logical_and(tr_ind, weight > 0))
        tran.reflect(int_rays,ind = logical_and(tr_ind, weight > 0))
        if xou.ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            weight *= ArcPerf.ref_weighting_ind(int_rays,wavelength,xou.ref_func,ind = logical_and(tr_ind, weight > 0))
        
        
        surf.spoSecondary(int_rays,r,xou.z0,ind = logical_and(tr_ind, weight > 0))
        tran.reflect(int_rays,ind = logical_and(tr_ind, weight > 0))
        if xou.ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            weight *= ArcPerf.ref_weighting_ind(int_rays,wavelength,xou.ref_func,ind = logical_and(tr_ind, weight > 0))
        

    # Add normalized scatter to the direction cosines if desired.
    if xou.scatter[0] == 'Gaussian':
        int_rays[4] = int_rays[4] + random.normal(scale=xou.dispdir_scatter_val,size=shape(int_rays)[1])
    else:
        print "You have selected a scatter profile that is not enabled. You will need to employ the (depreciated) ScatterDraw module." 
        pdb.set_trace()

    if xou.scatter[1] == 'Gaussian':
        int_rays[5] = int_rays[5] + random.normal(scale=xou.crossdispdir_scatter_val,size=shape(int_rays)[1])
    else:
        print "You have selected a scatter profile that is not enabled. You will need to employ the (depreciated) ScatterDraw module." 
        pdb.set_trace()

    int_rays[6] = -sqrt(1.- int_rays[5]**2 - int_rays[4]**2)
    
    # Now we apply the vignetting to all the PyXFocus rays, and undo the original
    # coordinate transformation (i.e. undoing the transform to the xou_coords done
    # at the start of this function)
    reref_xou_rays = ArcUtil.undo_ray_transform_to_coordinate_system(int_rays,xou.xou_coords)
    # Finally, reconstructing a vignetted ray object with the correctly tracked parameters, and
    # setting the ray objects PyXFocus rays to be those traced here.
    
    xou_ray_object = ray_object.yield_object_indices(ind = ones(len(reref_xou_rays[0]),dtype = bool))
    xou_ray_object.set_prays(reref_xou_rays)
    return xou_ray_object

def SPOPetalTrace(ray_object,xou_dict):
    # Identifying the XOUs hit for all of the input rays.
    id_XOU_for_rays(ray_object,xou_dict)
    # print sum(ray_object.xou_hit >= 0)
    petal_ray_dict = dict()
    # Looping through the entire dictionary of XOUs. 
    for key in xou_dict.keys():
        ray_ind_this_xou = ray_object.xou_hit == xou_dict[key].xou_num
        xou_ray_object = ray_object.yield_object_indices(ind = ray_ind_this_xou)
        if size(xou_ray_object.x) != 0:
            try:
                petal_ray_dict[key] = XOUTrace(xou_ray_object,xou_dict[key])
            except:
                pdb.set_trace()
                continue

    missed_rays = ray_object.yield_object_indices(ind = logical_or(isnan(ray_object.xou_hit), \
        ray_object.weight == 0))
    missed_rays.weight *= 0
    petal_ray_dict['XOU Miss'] = missed_rays

    petal_ray_object = ArcRays.merge_ray_object_dict(petal_ray_dict)
 
    return petal_ray_object
