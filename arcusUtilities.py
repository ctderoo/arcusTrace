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

##########################################################################
# Functions needed.
##########################################################################

def make_rot_matrix(tgrat,pgrat,ngrat):
    return transpose([tgrat,pgrat,ngrat])

def return_euler_angles(coord_sys):
    R = make_rot_matrix(coord_sys.xhat,coord_sys.yhat,coord_sys.zhat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    return a1,a2,a3

def do_ray_transform_to_coordinate_system(rays,coord_sys):
    transform_rays = copy_rays(rays)
    R = make_rot_matrix(coord_sys.xhat,coord_sys.yhat,coord_sys.zhat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    tran.transform(transform_rays,coord_sys.x,coord_sys.y,coord_sys.z,a1,a2,a3)
    #pdb.set_trace()
    # return [asarray(transform_rays)[i] for i in range(len(rays))]
    return asarray(transform_rays)

def undo_ray_transform_to_coordinate_system(rays,coord_sys):
    transform_rays = copy_rays(rays)
    R = make_rot_matrix(coord_sys.xhat,coord_sys.yhat,coord_sys.zhat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    tran.transform(transform_rays,0,0,0,0, 0,-a3)
    tran.transform(transform_rays,0,0,0,0,-a2,0)
    tran.transform(transform_rays,0,0,0,-a1,0,0)
    tran.transform(transform_rays,-coord_sys.x,-coord_sys.y,-coord_sys.z,0,0,0)
    #pdb.set_trace()
    # return [asarray(transform_rays)[i] for i in range(len(rays))]
    return asarray(transform_rays)

def chan_to_instrum_transform(transform_rays,coord_sys,refx = False,refy = False,inverse = False):
    rays = copy_rays(transform_rays)
    ref_mat = tran.tr.identity_matrix()[:3,:3]
    
    if inverse == True:
        rays = do_ray_transform_to_coordinate_system(rays,coord_sys)
    
        if refy == True:
            ref_mat = dot(ref_mat,tran.tr.reflection_matrix(array([0,0,0]),array([0,1,0]))[:3,:3])
        if refx == True:
            ref_mat = dot(ref_mat,tran.tr.reflection_matrix(array([0,0,0]),array([1,0,0]))[:3,:3])
            
        locs,vecs = vstack((rays[1],rays[2],rays[3])),vstack((rays[4],rays[5],rays[6]))
        new_locs = dot(ref_mat,locs)
        new_vecs = dot(ref_mat,vecs)
        rays[1],rays[2],rays[3],rays[4],rays[5],rays[6] = new_locs[0],new_locs[1],new_locs[2],new_vecs[0],new_vecs[1],new_vecs[2]
        
        moved_rays = copy_rays(rays)
    
    else:
        if refy == True:
            ref_mat = dot(ref_mat,tran.tr.reflection_matrix(array([0,0,0]),array([0,1,0]))[:3,:3])
        if refx == True:
            ref_mat = dot(ref_mat,tran.tr.reflection_matrix(array([0,0,0]),array([1,0,0]))[:3,:3])
            
        locs,vecs = vstack((rays[1],rays[2],rays[3])),vstack((rays[4],rays[5],rays[6]))
        new_locs = dot(ref_mat,locs)
        new_vecs = dot(ref_mat,vecs)
        rays[1],rays[2],rays[3],rays[4],rays[5],rays[6] = new_locs[0],new_locs[1],new_locs[2],new_vecs[0],new_vecs[1],new_vecs[2]
        
        moved_rays = undo_ray_transform_to_coordinate_system(rays,coord_sys)
    return moved_rays
    
####################################################################
# Utility-related functions.

def copy_rays(rays):
    return [rays[i].copy() for i in range(len(rays))]

####################################################################
# Performance-related functions.

def compute_cdf_pdf(rays1d):
    hist,bin_edges = histogram(rays1d,bins = 10**3)
    x_locs = (bin_edges[:-1] + bin_edges[1:])/2
    pdf = hist.astype(float)/sum(hist)
    cdf = cumsum(hist).astype(float)/sum(hist)
    return cdf,pdf,x_locs
 
def hpd_cdf_intersection(cdf,x_locs):
    ind1,ind2 = argmin(abs(cdf - 0.25)),argmin(abs(cdf - 0.75))
    return x_locs[ind2] - x_locs[ind1]

def compute_hpd(rays1d):
    cdf,pdf,x_locs = compute_cdf_pdf(rays1d)
    return hpd_cdf_intersection(cdf,x_locs)

def compute_FWHM(rays1d):
    '''
    Based on Gaussian scaling of 1.35 sigma width for containing HPD, 2.35 sigma width for FWHM
    '''
    return compute_hpd(rays1d)*2.35/1.35

#############################################
# Ray picking functions. 

def select_xou_rays(rays,xou_hit,ind):
    return asarray(rays)[:,where(xou_hit == ind)[0]]

def select_grat_rays(rays,grat_hit,ind):
    return asarray(rays)[:,where(grat_hit == ind)[0]]

def check_normals(rays,grat_hit,ind):
    sample_rays = select_grat_rays(rays,ind)
    norms = vstack((sample_rays[4],sample_rays[5],sample_rays[6]))
    norms = transpose(norms)
    ngrat = cfpar.ngrats[ind]
    return norms,ngrat

def get_xou_ray_object(ray_object,ind):
    selector = ray_object.xou_hit == ind
    
