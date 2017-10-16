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
import PyXFocus.transformMod as transM
import PyXFocus.analyses as anal

import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.ParamFiles.arcus_params_rev1p7 as cfpar

####################################################################
# SPO-Related Functions
####################################################################

def SPO_ray_select(rays,rin,rout,width,clock_angle):
    '''
    Looks up the rays that will hit a given SPO.
    '''
    clocked_rays = tran.copy_rays(rays)
    tran.transform(clocked_rays,0.,0.,0.,0.,0.,-clock_angle*pi/180)
    clocked_rays = asarray(clocked_rays)
    radii = sqrt(clocked_rays[1]**2 + clocked_rays[2]**2)
    ind = logical_and(logical_and(radii > rin,radii < rout),abs(clocked_rays[1]) <= width/2)
    return [asarray(rays)[i][ind] for i in range(len(rays))]

def SPOtrace(chan_rays,rin=700.,rout=737.,azwidth=66.,F = 12000.,\
             scatter=True,ref_func = None):
    """
    Trace a set of rays through an SPO module using a
    finite source distance. Incorporate plate-by-plate
    vignetting.
    Set up a subannulus, apply finite source effect,
    and then simply translate to inner SPO radius and
    trace outward.
    Inputs:
    N = number of rays
    rin = inner radius of the SPO module (mm)
    rout = outer radius of the SPO module (mm)
    azwidth = linear width of the SPO module (mm)
    F = nominal focal length of the SPO module
    scatter = boolean setting whether or not the SPO should scatter to be consistent with 2" HPD dispersion, 10" HPD cross-dispersion
    Outputs:
    rays = a ray object centered on the SPO, with:
        x - the radial coordinates, zeroed at the center of the SPO radius,
        y - the azimuthal coordinates, zeroed at the center of the SPO module
        z - the axial coordinate, with positive z pointed towards the source,
        and zeroed at the center of the SPO interface.
    """
    plate_height = 0.775
    pore_space = 0.605
    
    # Define the angular ranges of the SPOs.
    R = arange(rin,rout,plate_height)
    rad = sqrt(chan_rays[1]**2+chan_rays[2]**2)
    v_ind_all = zeros(len(chan_rays[1]),dtype = bool)
    # Bookkeeping variables in case checking how many rays get vignetted (and through what process) is necessary.
    geo_v,ref_v = 0,0
    
    for r in R:
        # Collect relevant rays: tr_ind are the rays that should be traced,
        # v_ind are the rays that should be vignetted (impacting plates).
        # Note that the sidewall vignetting is completely ignored.
        tr_ind = logical_and(rad > r,rad < r + pore_space)
        v_ind = logical_and(rad > r + pore_space, rad < r + plate_height)
        geo_v = geo_v + sum(v_ind)
        v_ind_all = logical_or(v_ind,v_ind_all)
        
        # In case our sampling is so sparse that there are no rays fulfilling this condition.
        if sum(tr_ind) == 0: 
            continue
        
        surf.spoPrimary(chan_rays,r,F,ind=tr_ind)
        tran.reflect(chan_rays,ind = tr_ind)
        if ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            v_ind_ref = ArcPerf.ref_vignette_ind(chan_rays,ref_func,ind = tr_ind)
            ref_v = ref_v + sum(v_ind_ref)
            v_ind_all = logical_or(v_ind_ref,v_ind_all)
        
        surf.spoSecondary(chan_rays,r,F,ind = tr_ind)
        tran.reflect(chan_rays,ind = tr_ind)
        if ref_func is not None:
            # Calculating the rays lost to absorption. This enters as vignetting after passing through the entire SPO.
            v_ind_ref = ArcPerf.ref_vignette_ind(chan_rays,ref_func,ind = tr_ind)
            ref_v = ref_v + sum(v_ind_ref)
            v_ind_all = logical_or(v_ind_ref,v_ind_all)
    
    # Rays hitting the SPO mirror are now at secondary surfaces.
    # Vignette the rays hitting the plate stacks.
    chan_rays = tran.vignette(chan_rays,ind = ~v_ind_all)
    
    # Add normalized scatter to the direction cosines if desired.
    if scatter is True:
        chan_rays[4] = chan_rays[4] + random.normal(scale=1.5/2.35*5e-6,size=shape(chan_rays)[1])
        chan_rays[5] = chan_rays[5] + random.normal(scale=15./2.35*5e-6,size=shape(chan_rays)[1])
        chan_rays[6] = -sqrt(1.-chan_rays[5]**2-chan_rays[4]**2)
    return chan_rays

def traceSPOFocus(N,F = 12000.,rin=700.,rout=737.,azwidth=66,scatter = True):
    subrays = source.subannulus(rin,rout,azwidth/rin,N,zhat=-1.)
    tran.transform(subrays,0,0,0,0,0,-pi/2)
    sub2rays = SPOtrace(subrays,scatter = scatter)
    tran.transform(sub2rays,0.,0.,-cfpar.F,0.,0.,0.)
    surf.flat(sub2rays)
    return sub2rays,subrays

def slope_trans(rays,angle):
    rot_mat = transM.rotation_matrix(angle,array([1,0,0,1]))
    slopes = vstack((rays[4],rays[5],rays[6],ones(len(rays[4]))))
    new_slopes = dot(rot_mat,slopes)
    rays[4],rays[5],rays[6] = new_slopes[0,:],new_slopes[1,:],new_slopes[2,:]
    return

def SPOPetalTrace(rays,module_range = range(cfpar.N_xous),apply_reflectivity = True,scatter = True):
    if apply_reflectivity == True:
        # A one-time call to set up the reflectivity function for this SPO XOU.
        ref_func = ArcPerf.make_reflectivity_func(cfpar.MM_coat_mat,cfpar.MM_coat_rough)
    else:
        ref_func = None
        
    for i in module_range:
        # Produces the selection of rays hitting a particular (the "ith") XOU.
        xou_rays = SPO_ray_select(rays,cfpar.xou_irs[i],cfpar.xou_ors[i],cfpar.xou_widths[i],cfpar.xou_cangles[i])
        
        # Clocking the rays at the SPO clock angle.
        clocked_rays = tran.copy_rays(xou_rays)
        tran.transform(clocked_rays,0.,0.,0.,0.,0.,-cfpar.xou_cangles[i]*pi/180)

        # Performing the trace through the SPO under `square on' incidence conditions, where the rays may be at a tilt set
        # by the cfpar.xou_tilts. 
        thru_rays = SPOtrace(clocked_rays,rin = cfpar.xou_irs[i],rout = cfpar.xou_ors[i],azwidth = cfpar.xou_widths[i],\
                             scatter = scatter,ref_func = ref_func)
        
        # Reversing the clock angle and tracing the rays to the focal plane.
        tran.transform(thru_rays,0.,0.,-cfpar.F,0.,0.,cfpar.xou_cangles[i]*pi/180)
        
        if i == module_range[0]:
            spo_rays = thru_rays
            xou_hit = ones(len(spo_rays[0]))*i
        else:
            spo_rays = hstack((spo_rays,thru_rays))
            xou_hit = hstack((xou_hit,ones(len(thru_rays[0]))*i))
    return spo_rays,xou_hit