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
import arcusTrace.ParamFiles.arcus_params_rev1p8 as cfpar

####################################################################
# Detector-related functions.

def make_rot_matrix(tgrat,pgrat,ngrat):
    return transpose([tgrat,pgrat,ngrat])

def check_size(rays,phy_size):
    x,y = rays[1],rays[2]
    xcond = logical_and(rays[1] > -phy_size[0]/2,rays[1] < phy_size[0]/2)
    ycond = logical_and(rays[2] > -phy_size[1]/2,rays[2] < phy_size[1]/2)
    tcond = logical_and(xcond,ycond)
    return tcond

#def define_det_array(xlocs,RoC):
#    '''
#    From the channel origin, explicitly defines the detector array in the plane y = 0
#    from the x dimension locations specified. Number of detectors is given by the length
#    of the location array.
#    ''' 
#    zlocs = RoC - sqrt(RoC**2 - xlocs**2)
#    locs = array([array([xlocs[i],0,zlocs[i]]) for i in range(len(xlocs))])
#    
#    normals = array([array([-xlocs[i],0,RoC - zlocs[i]])/RoC for i in range(len(xlocs))])
#    cross_disp_dir = array([array([0,1,0]) for i in range(len(xlocs))])
#    disp_dir = array([cross(cross_disp_dir[i],normals[i]) for i in range(len(xlocs))])
#    
#    det_vecs = array([vstack((disp_dir[i],cross_disp_dir[i],normals[i])) for i in range(len(xlocs))])
#    return locs,det_vecs

def single_ccd(rays,loc,norm,ccd_size = (50,25)):
    '''
    Propagate rays to a single CCD.
    Inputs:
    rays -- rays to be propagated.
    loc -- 3D position of the center of the detector array.
    norm -- normal of the CCD face
    ccd_size -- X,Y dimensions of the CCD.
    Outputs:
    good_rays -- rays on the detector
    ind -- the indices of the rays caught by this CCD from the original input rays
    '''
    det_rays = ArcUtil.copy_rays(rays)
        
    det_x,det_y,det_z = loc
    det_nx,det_ny,det_nz = norm
    
    R = make_rot_matrix(det_nx,det_ny,det_nz)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(det_rays,det_x,det_y,det_z,a1,a2,a3)
    surf.flat(det_rays)
    
    ind = check_size(det_rays,ccd_size)
    good_rays = tran.vignette(det_rays,ind)
    
    # If the detector is empty, return the empty ray list.
    if sum(ind) == 0:
        return good_rays,ind
    
    tran.transform(good_rays,0,0,0,0,0,-a3)
    tran.transform(good_rays,0,0,0,0,-a2,0)
    tran.transform(good_rays,0,0,0,-a1,0,0)
    tran.transform(good_rays,-det_x,-det_y,-det_z,0,0,0)
    return good_rays,ind

def DetectorArrayTrace(rays,diff_order,grat_hit,apply_qe = True,apply_contam = True,\
                       apply_filters = True,det_locs = cfpar.det_locs,det_norms = cfpar.det_vecs):
    
    def vignette_ray_stats(ray_stats,good_ind):
        return ray_stats[good_ind]
    
    for i in range(len(det_locs)):
        ccd_rays,ccd_ind = single_ccd(rays,det_locs[i],det_norms[i])
        if i == 0:
            det_rays = ccd_rays
            det_ind = ccd_ind
            det_hit = ones(len(ccd_rays[0]))*i
            det_ray_order = diff_order[ccd_ind]
            det_ray_grat_hit = grat_hit[ccd_ind]
        else:
            det_rays = hstack((det_rays,ccd_rays))
            det_ind = logical_or(det_ind,ccd_ind)
            det_hit = hstack((det_hit,ones(len(ccd_rays[0]))*i))
            det_ray_order = hstack((det_ray_order,diff_order[ccd_ind]))
            det_ray_grat_hit = hstack((det_ray_grat_hit,grat_hit[ccd_ind]))
    #print len(det_rays[0]),sum(det_ind)
    
    if apply_qe == True:
        det_rays,qe_vig_list = ArcPerf.apply_detector_effect_vignetting(det_rays,ArcPerf.det_qe_fn)
        det_hit = det_hit[qe_vig_list]
        det_ray_order = det_ray_order[qe_vig_list]
        det_ray_grat_hit = det_ray_grat_hit[qe_vig_list]
        
    if apply_contam == True:
        det_rays,contam_vig_list = ArcPerf.apply_detector_effect_vignetting(det_rays,ArcPerf.det_contam_fn)
        det_hit = det_hit[contam_vig_list]
        det_ray_order = det_ray_order[contam_vig_list]
        det_ray_grat_hit = det_ray_grat_hit[contam_vig_list]
    
    if apply_filters == True:
        det_rays,opt_filter_vig_list = ArcPerf.apply_detector_effect_vignetting(det_rays,ArcPerf.opt_block_fn)
        det_hit = det_hit[opt_filter_vig_list]
        det_ray_order = det_ray_order[opt_filter_vig_list]
        det_ray_grat_hit = det_ray_grat_hit[opt_filter_vig_list]

        det_rays,uv_filter_vig_list = ArcPerf.apply_detector_effect_vignetting(det_rays,ArcPerf.uv_block_fn)
        det_hit = det_hit[uv_filter_vig_list]
        det_ray_order = det_ray_order[uv_filter_vig_list]
        det_ray_grat_hit = det_ray_grat_hit[uv_filter_vig_list]
        
        det_rays,si_mesh_vig_list = ArcPerf.apply_detector_effect_vignetting(det_rays,ArcPerf.Si_mesh_block_fn)
        det_hit = det_hit[si_mesh_vig_list]
        det_ray_order = det_ray_order[si_mesh_vig_list]
        det_ray_grat_hit = det_ray_grat_hit[si_mesh_vig_list]
    
    return det_rays,det_hit,det_ray_order,det_ray_grat_hit