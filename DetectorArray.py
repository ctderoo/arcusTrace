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
import arcusTrace.ParamFiles.arcus_params_rev1p7 as cfpar

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

def define_det_array(xlocs,RoC):
    '''
    From the channel origin, explicitly defines the detector array in the plane y = 0
    from the x dimension locations specified. Number of detectors is given by the length
    of the location array.
    ''' 
    zlocs = RoC - sqrt(RoC**2 - xlocs**2)
    locs = array([array([xlocs[i],0,zlocs[i]]) for i in range(len(xlocs))])
    
    normals = array([array([-xlocs[i],0,RoC - zlocs[i]])/RoC for i in range(len(xlocs))])
    cross_disp_dir = array([array([0,1,0]) for i in range(len(xlocs))])
    disp_dir = array([cross(cross_disp_dir[i],normals[i]) for i in range(len(xlocs))])
    
    det_vecs = array([vstack((disp_dir[i],cross_disp_dir[i],normals[i])) for i in range(len(xlocs))])
    return locs,det_vecs

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
    if sum(ind) == 0:
        pdb.set_trace()
    
    tran.transform(good_rays,0,0,0,0,0,-a3)
    tran.transform(good_rays,0,0,0,0,-a2,0)
    tran.transform(good_rays,0,0,0,-a1,0,0)
    tran.transform(good_rays,-det_x,-det_y,-det_z,0,0,0)
    return good_rays,ind

def DetectorArrayTrace(rays):
    det_locs,det_norms = define_det_array(cfpar.det_xlocs,cfpar.det_RoC)
    
    for i in range(len(det_locs)):
        ccd_rays,ccd_ind = single_ccd(rays,det_locs[i],det_norms[i])
        if i == 0:
            det_rays = ccd_rays
            det_ind = ccd_ind
            det_hit = ones(len(ccd_rays[0]))*i
        else:
            det_rays = hstack((det_rays,ccd_rays))
            det_ind = hstack((det_ind,ccd_ind))
            det_hit = hstack((det_hit,ones(len(ccd_rays[0]))*i))
    return det_rays,det_ind,det_hit