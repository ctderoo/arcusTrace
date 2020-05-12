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
# CAT Grating Related Functions

def id_facet_for_rays(ray_object,facet_dict):
    rays = ray_object.yield_prays()
    facet_hit = empty(len(rays[0]))
    facet_hit.fill(NaN)
    
    for key in facet_dict.keys():
        grat_rays = ArcUtil.do_ray_transform_to_coordinate_system(rays,facet_dict[key].facet_coords)
        surf.flat(grat_rays)
        bool_list = (grat_rays[1] < facet_dict[key].xsize/2)*(grat_rays[1] > -facet_dict[key].xsize/2)*\
            (grat_rays[2] < facet_dict[key].ysize/2)*(grat_rays[2] > -facet_dict[key].ysize/2)*\
            (ray_object.weight > 0)
        facet_hit[bool_list] = facet_dict[key].facet_num
    ray_object.facet_hit = facet_hit

def grat_ray_select(rays,xgrat,ygrat,zgrat,tgrat,pgrat,ngrat,xwidth,ywidth):
    grat_rays = ArcUtil.copy_rays(rays)
    R = make_rot_matrix(tgrat,pgrat,ngrat)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
    surf.flat(grat_rays)
    
    ind = (grat_rays[1] < xwidth/2)*(grat_rays[1] > -xwidth/2)*(grat_rays[2] < ywidth/2)*(grat_rays[2] > -ywidth/2)
    return [asarray(rays)[i][ind] for i in range(len(rays))]

def prop_to_grat(rays,xgrat,ygrat,zgrat,normgrat):
    # Loosely based on:
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    
    grat_rays = ArcUtil.copy_rays(rays)
    
    zhat = array([0,0,1])
    cvec,ctheta = cross(zhat,normgrat),dot(zhat,normgrat)
    skew_mat = tran.skew(cvec)
    I = array([[1,0,0],[0,1,0],[0,0,1]])
    R = I + skew_mat + dot(skew_mat,skew_mat)*1/(1 + ctheta)
    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
    surf.flat(grat_rays)
    tran.transform(grat_rays,0,0,0,0, 0,-a3)
    tran.transform(grat_rays,0,0,0,0,-a2,0)
    tran.transform(grat_rays,0,0,0,-a1,0,0)
    tran.transform(grat_rays,-xgrat,-ygrat,-zgrat,0,0,0)
    return grat_rays

def make_rot_matrix(tgrat,pgrat,ngrat):
    return transpose([tgrat,pgrat,ngrat])

# Is this not needed? It's the only function that calls cfpar, which makes me think it's depreciated. 5/5/20.
#def CATgratTrace(rays,order,xgrat,ygrat,zgrat,tgrat,pgrat,ngrat,dgrat = 2.00e-4):
#    '''
#    Loosely based on: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
#    '''
#    grat_rays = ArcUtil.copy_rays(rays)
#    R = make_rot_matrix(tgrat,pgrat,ngrat)
#    a1,a2,a3 = tran.tr.euler_from_matrix(R,'sxyz')
    
#    tran.transform(grat_rays,xgrat,ygrat,zgrat,a1,a2,a3)
#    surf.flat(grat_rays)
    
#    tran.grat(grat_rays,dgrat,order,grat_rays[0])
#    tran.reflect(grat_rays)
    
#    tran.transform(grat_rays,0,0,0,0, 0,-a3)
#    tran.transform(grat_rays,0,0,0,0,-a2,0)
#    tran.transform(grat_rays,0,0,0,-a1,0,0)
#    tran.transform(grat_rays,-xgrat,-ygrat,-zgrat,0,0,0)
#    return grat_rays

#    for j in range(cfpar.N_grats):
#        # Selecting the rays hitting the jth grating and creating a separate selection of "single grating rays" -- sgrat_rays
#        sgrat_rays = grat_ray_select(rays,cfpar.xgrats[j],cfpar.ygrats[j],cfpar.zgrats[j],\
#                                                cfpar.tgrats[j],cfpar.pgrats[j],cfpar.ngrats[j],cfpar.grat_dims[0],cfpar.grat_dims[1])
    
        
#        #pdb.set_trace()
#        # Now tracking the output of diffracted rays (diff_rays) and which ray hits which grating. This effectively "auto-vignettes"
#        # rays that don't pass through a grating, so some care is required.
#        if j == 0:
#            diff_rays = sdiff_rays
#            diff_orders = order
#            grat_hit = ones(len(sdiff_rays[0]))*j
#        else:
#            diff_rays = hstack((diff_rays,sdiff_rays))
#            diff_orders = hstack((diff_orders,order))
#            grat_hit = hstack((grat_hit,ones(len(sdiff_rays[0]))*j))
#    return diff_rays,diff_orders,grat_hit

def GratFacetTrace(ray_object,facet):
    # Getting the PyXFocus rays from the object.
    init_rays = ray_object.yield_prays()
    orders = ray_object.order
    wavelengths = ray_object.wave
    weight = ray_object.weight
    
    # Performing the grating structure vignetting.
    if facet.L1supp == True:
        weight *= ArcPerf.apply_support_weighting(init_rays,support = 'L1')
    if facet.L2supp == True:
        weight *= ArcPerf.apply_support_weighting(init_rays,support = 'L2')

    # Multiplying in the order-weighted Debye-Waller factor.
    if facet.debye_waller == True:
        weight *= ArcPerf.apply_debye_waller_weighting(orders)

    # Negative sign necessary due to definition of grating normal away from telescope focus, and ray direction towards the focus.
    thetas = anal.indAngle(init_rays,normal = -facet.facet_coords.zhat)*180/pi
    if facet.diff_eff == True:
        weight *= facet.geff_func((wavelengths,thetas,orders))
    
    # If there are rays hitting the grating, do the transform and diffraction. Otherwise, pass the
    # the empty ray object.
    try:
        # Tracing the selected rays to the grating itself.
        facet_rays = ArcUtil.do_ray_transform_to_coordinate_system(init_rays,facet.facet_coords)
        surf.flat(facet_rays)
        # Actually performing the diffraction on the grating.    
        tran.grat(facet_rays,facet.period,orders,wavelengths)
        tran.reflect(facet_rays)
    except:
        facet_rays = init_rays

    # If needed, adding any additional grating scatter in the dispersion direction.
    if facet.grat_scatter_val is not None:
        facet_rays[4] = facet_rays[4] + random.normal(scale=facet.grat_scatter_val,size=shape(facet_rays)[1])
        facet_rays[6] = -sqrt(1. - facet_rays[4]**2 - facet_rays[5]**2)

    # Now we clean-up the ray object -- we specified the order and calculated theta, but 
    # now carry those characteristics forward only for rays with weight > 0 i.e. rays that 
    # weren't vignetted in this process.
    transmitted_orders = zeros(len(orders))
    transmitted_orders[weight > 0] = orders[weight > 0]
    theta_on_grat = full(len(thetas),nan)
    theta_on_grat[weight > 0] = thetas[weight > 0]
    
    # If there are rays left, transform them back. If there are no more rays, don't transform -- just use the blank ray object.
    try:
        reref_facet_rays = ArcUtil.undo_ray_transform_to_coordinate_system(facet_rays,facet.facet_coords)
    except:
        print 'No rays on the facet -- skipping...'
        reref_facet_rays = facet_rays
        
    # Finally, reconstructing a vignetted ray object with the correctly tracked parameters, and
    # setting the ray objects PyXFocus rays to be those traced here.
    facet_ray_object = ray_object.yield_object_indices(ind = ones(len(facet_rays[0]),dtype = bool))
    facet_ray_object.set_prays(reref_facet_rays)

    # Adding two new attributes to our ray object:
    # an order tracker and the angle of incidence on the grating.
    if len(transmitted_orders) != len(facet_ray_object.x):
        pdb.set_trace()
   
    facet_ray_object.order = transmitted_orders
    facet_ray_object.theta_on_facet = theta_on_grat
    
    return facet_ray_object

def CATPetalTrace(ray_object,facet_dict):
    id_facet_for_rays(ray_object,facet_dict)
    
    facet_ray_dict = dict()
    # Looping through the entire dictionary of XOUs. 
    for key in facet_dict.keys():
        ray_ind_this_facet = ray_object.facet_hit == facet_dict[key].facet_num
        facet_ray_object = ray_object.yield_object_indices(ind = ray_ind_this_facet)
        facet_ray_dict[key] = GratFacetTrace(facet_ray_object,facet_dict[key])

    missed_rays = ray_object.yield_object_indices(ind = logical_or(isnan(ray_object.facet_hit), \
        ray_object.weight == 0))
    missed_rays.weight *= 0
    missed_rays.theta_on_facet = full(len(missed_rays.x),nan)
    missed_rays.order = zeros(len(missed_rays.x))
    facet_ray_dict['CAT Miss'] = missed_rays
            
    facet_ray_object = ArcRays.merge_ray_object_dict(facet_ray_dict)
    return facet_ray_object