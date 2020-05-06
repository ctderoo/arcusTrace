from numpy import *
import matplotlib.pyplot as plt
import os
import pdb
import pickle

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic

import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusRays as ArcRays

####################################################################
# Detector-related functions.
    
def check_size(rays,phy_size):
    '''
    Checking that a given ray will hit a CCD given the coordinate transformation and the physical size
    of the CCD.
    '''
    x,y = rays[1],rays[2]
    xcond = logical_and(rays[1] > 0,rays[1] < phy_size[0])
    ycond = logical_and(rays[2] > 0,rays[2] < phy_size[1])
    tcond = logical_and(xcond,ycond)
    return tcond

def id_ccd_for_rays(ray_object,ccd_dict):
    rays = ray_object.yield_prays()
    ccd_hit = empty(len(rays[0]))
    ccd_hit.fill(NaN)
    
    for key in ccd_dict.keys():
        copy_rays = ArcUtil.copy_rays(rays)
        ccd_plane_rays = ArcUtil.do_ray_transform_to_coordinate_system(copy_rays,ccd_dict[key].ccd_coords)
        surf.flat(ccd_plane_rays)
        bool_list = logical_and(ray_object.weight > 0, check_size(ccd_plane_rays, (ccd_dict[key].xwidth,ccd_dict[key].ywidth)))
        ccd_hit[bool_list] = ccd_dict[key].ccd_num
    ray_object.ccd_hit = ccd_hit

def ccdTrace(ray_object,ccd):
    '''
    Propagate rays to a single CCD.
    Inputs:
    rays -- rays to be propagated.
    loc -- 3D position of the lower left corner (pixel 0,0) of the detector array.
    norm -- normal of the CCD face
    ccd_size -- X,Y dimensions of the CCD.
    Outputs:
    good_rays -- rays on the detector
    ind -- the indices of the rays caught by this CCD from the original input rays
    '''
    init_rays = ray_object.yield_prays()
    wavelength = ray_object.wave
    weight = ray_object.weight

    # v_ind_all = zeros(len(init_rays[0]),dtype = bool)
    
    # Performing all the vignetting due to detector effects.
    for i in range(len(ccd.det_effects)):
        weight *= ArcPerf.apply_detector_effect_weighting(init_rays,wavelength,ccd.det_effects[i].filter_func)
    
    # Implemented hot fix on 2/14/19 -- mismatch between length of order vector vs. position vector. Stems from a problem 
    # of assignment on the detector side -- if a handful of rays are vignetted by the CCD vignetting, the ArcUtil.undo_transform
    # function throws an error, causing us to assign the ray object the position of the original rays (nonzero length) rather than the vignetted (zero length)
    # rays. However, the order attribute is properly vignetted to zero length. This later results in an error due to the mismatch of the order vector vs. the position vector.
    
    try: 
        ccd_rays = ArcUtil.do_ray_transform_to_coordinate_system(init_rays,ccd.ccd_coords)
        surf.flat(ccd_rays)
    except:
        pdb.set_trace()
    # Per comments on line 68, this should result in the same length order vs. position vector.
    try:
        reref_ccd_rays = ArcUtil.undo_ray_transform_to_coordinate_system(ccd_rays,ccd.ccd_coords)
    except:
        reref_ccd_rays = ccd_rays
    
    # Finally, reconstructing a vignetted ray object with the correctly tracked parameters, and
    # setting the ray objects PyXFocus rays to be those traced here.
    ccd_ray_object = ray_object.yield_object_indices(ind = ones(len(reref_ccd_rays[0]),dtype = bool))
    ccd_ray_object.set_prays(reref_ccd_rays)
    
    if len(ccd_ray_object.x) != len(reref_ccd_rays[1]):
        pdb.set_trace()

    return ccd_ray_object

def DetectorArrayTrace(ray_object,ccd_dict):
    # Identifying the XOUs hit for all of the input rays.    
    id_ccd_for_rays(ray_object,ccd_dict)
    
    det_ray_dict = dict()
    # Looping through the entire dictionary of XOUs. 
    for key in ccd_dict.keys():
        ray_ind_this_ccd = ray_object.ccd_hit == ccd_dict[key].ccd_num
        # Handling the case where there are no rays on this CCD.
        if sum(ray_ind_this_ccd) == 0:
            continue
        else:
            ccd_ray_object = ray_object.yield_object_indices(ind = ray_ind_this_ccd)
            try: 
                det_ray_dict[key] = ccdTrace(ccd_ray_object,ccd_dict[key])
            except:
                pdb.set_trace()

    missed_rays = ray_object.yield_object_indices(ind = logical_or(isnan(ray_object.ccd_hit), \
        ray_object.weight == 0))
    missed_rays.weight *= 0
    det_ray_dict['Detector Miss'] = missed_rays
            
    ccd_ray_object = ArcRays.merge_ray_object_dict(det_ray_dict)
    return ccd_ray_object