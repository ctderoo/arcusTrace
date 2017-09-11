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

import arcusTrace.ParamFiles.arcus_params_rev1p4 as cfpar

####################################################################
# SPO-Related Functions

def SPO_ray_select(rays,rin,rout,width,clock_angle):
    '''
    Looks up the rays that will hit a given SPO.
    '''
    clocked_rays = copy_rays(rays)
    tran.transform(clocked_rays,0.,0.,0.,0.,0.,-clock_angle*pi/180)
    clocked_rays = asarray(clocked_rays)
    radii = sqrt(clocked_rays[1]**2 + clocked_rays[2]**2)
    ind = logical_and(logical_and(radii > rin,radii < rout),abs(clocked_rays[1]) <= width/2)
    return [asarray(rays)[i][ind] for i in range(len(rays))]

def SPOtrace(chan_rays,rin=700.,rout=737.,azwidth=66.,F = 12000.,\
             scatter=True):
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
    srcdist = Distance from the SPO interface to the X-ray source (mm)
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
    for r in R:
        # Collect relevant rays: tr_ind are the rays that should be traced,
        # v_ind are the rays that should be vignetted (impacting plates).
        # Note that the sidewall vignetting is completely ignored.
        tr_ind = logical_and(rad > r,rad < r + pore_space)
        v_ind = logical_and(rad > r + pore_space, rad < r + plate_height)
        v_ind_all = logical_or(v_ind,v_ind_all)
        if sum(tr_ind) == 0: 
            continue
        surf.spoPrimary(chan_rays,r,F,ind=tr_ind)
        tran.reflect(chan_rays,ind = tr_ind)
        surf.spoSecondary(chan_rays,r,F,ind = tr_ind)
        tran.reflect(chan_rays,ind = tr_ind)    
    
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
    tran.transform(sub2rays,0.,0.,-F,0.,0.,0.)
    surf.flat(sub2rays)
    return sub2rays,subrays

def SPOPetalTrace(rays):
    for i in [0,1]:
        # Produces the selection of rays hitting a particular (the "ith") XOU.
        xou_rays = ray_ind_select(rays,cfpar.xou_loc_irs[i],cfpar.xou_loc_ors[i],cfpar.xou_widths[i],cfpar.xou_cangles[i])
        
        # Clocking the rays at the SPO clock angle.
        clocked_rays = copy_rays(xou_rays)
        tran.transform(clocked_rays,0.,0.,0.,0.,0.,-cfpar.xou_cangles[i]*pi/180)
    
        # Now moving the rays to concide with the optical axis of this XOU.
        tran.transform(clocked_rays,0,cfpar.xou_rad_offsets[i],0,cfpar.xou_tilts,0,0)
        
        # Performing the trace through the SPO under `square on' incidence conditions.
        thru_rays = SPOtrace(clocked_rays,rin = cfpar.xou_opt_irs[i],rout = cfpar.xou_opt_ors[i],azwidth = cfpar.xou_widths[i])
        
        # Undoing this transform to get to the XOU axis.
        tran.transform(thru_rays,0,0,0,-cfpar.xou_tilts[i],0,0)
        tran.transform(thru_rays,0,-cfpar.xou_rad_offsets[i],0,0,0,0)
        
        #pdb.set_trace()
        # Reversing the clock angle and tracing the rays to the focal plane.
        tran.transform(thru_rays,0.,0.,-F,0.,0.,cfpar.xou_cangles[i]*pi/180)
        
        if i == 0:
            spo_rays = thru_rays
            xou_hit = ones(len(spo_rays[0]))*i
        else:
            spo_rays = hstack((spo_rays,thru_rays))
            xou_hit = hstack((xou_hit,ones(len(thru_rays[0]))*i))
    return spo_rays,xou_hit