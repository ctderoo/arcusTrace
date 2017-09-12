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

import arcusTrace.ParamFiles.arcus_params_rev1p4 as cfpar

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
    tran.transform(sub2rays,0.,0.,-cfpar.F,0.,0.,0.)
    surf.flat(sub2rays)
    return sub2rays,subrays

def slope_trans(rays,angle):
    rot_mat = transM.rotation_matrix(angle,array([1,0,0,1]))
    slopes = vstack((rays[4],rays[5],rays[6],ones(len(rays[4]))))
    new_slopes = dot(rot_mat,slopes)
    rays[4],rays[5],rays[6] = new_slopes[0,:],new_slopes[1,:],new_slopes[2,:]
    return

#def SPOPetalTrace(rays):
#    for i in [0,1]:#range(cfpar.N_xous):
#        # Produces the selection of rays hitting a particular (the "ith") XOU.
#        xou_rays = SPO_ray_select(rays,cfpar.xou_loc_irs[i],cfpar.xou_loc_ors[i],cfpar.xou_widths[i],cfpar.xou_cangles[i])
#        
#        # Clocking the rays at the SPO clock angle.
#        clocked_rays = tran.copy_rays(xou_rays)
#        tran.transform(clocked_rays,0.,0.,0.,0.,0.,-cfpar.xou_cangles[i]*pi/180)
#    
#        pdb.set_trace()
#        # Now moving the rays to concide with the optical axis of this XOU.
#        tran.transform(clocked_rays,0,cfpar.xou_rad_offsets[i],0,cfpar.xou_tilts[i],0,0)
#
#        pdb.set_trace()
#        # Performing the trace through the SPO under `square on' incidence conditions, where the rays may be at a tilt set
#        # by the cfpar.xou_tilts. 
#        thru_rays = SPOtrace(clocked_rays,rin = cfpar.xou_opt_irs[i],rout = cfpar.xou_opt_ors[i],azwidth = cfpar.xou_widths[i])
#        
#        # Undoing this transform to get to the XOU axis.
#        tran.transform(thru_rays,0,0,0,-cfpar.xou_tilts[i],0,0)
#        tran.transform(thru_rays,0,-cfpar.xou_rad_offsets[i],0,0,0,0)
#        
#        # Reversing the clock angle and tracing the rays to the focal plane.
#        tran.transform(thru_rays,0.,0.,-cfpar.F,0.,0.,cfpar.xou_cangles[i]*pi/180)
#        
#        if i == 0:
#            spo_rays = thru_rays
#            xou_hit = ones(len(spo_rays[0]))*i
#        else:
#            spo_rays = hstack((spo_rays,thru_rays))
#            xou_hit = hstack((xou_hit,ones(len(thru_rays[0]))*i))
#    return spo_rays,xou_hit,clocked_rays

#def SPOPetalTrace(rays):
#    for i in [0,1]:#range(cfpar.N_xous):
#        # Produces the selection of rays hitting a particular (the "ith") XOU.
#        xou_rays = SPO_ray_select(rays,cfpar.xou_loc_irs[i],cfpar.xou_loc_ors[i],cfpar.xou_widths[i],cfpar.xou_cangles[i])
#        
#        # Clocking the rays at the SPO clock angle.
#        clocked_rays = tran.copy_rays(xou_rays)
#        tran.transform(clocked_rays,0.,0.,0.,0.,0.,-cfpar.xou_cangles[i]*pi/180)
#    
#        # Now moving the rays to concide with the optical axis of this XOU.
#        tran.transform(clocked_rays,0,cfpar.xou_rad_offsets[i],0,0,0,0)
#        
#        pdb.set_trace()
#        slope_trans(clocked_rays,cfpar.xou_tilts[i])
#        pdb.set_trace()
#        
#        # Performing the trace through the SPO under `square on' incidence conditions, where the rays may be at a tilt set
#        # by the cfpar.xou_tilts. Note that this rotation is about the origin (on optical axis, !NOT! about the module (so that )
#        thru_rays = SPOtrace(clocked_rays,rin = cfpar.xou_opt_irs[i],rout = cfpar.xou_opt_ors[i],azwidth = cfpar.xou_widths[i])
#        
#        # Undoing this transform to get to the XOU axis.
#        #slope_trans(thru_rays,cfpar.xou_tilts[i]*2)
#        tran.transform(thru_rays,0,-cfpar.xou_rad_offsets[i],0,0,0,0)
#        
#        # Reversing the clock angle and tracing the rays to the focal plane.
#        tran.transform(thru_rays,0.,0.,-cfpar.F,0.,0.,cfpar.xou_cangles[i]*pi/180)
#        
#        if i == 0:
#            spo_rays = thru_rays
#            xou_hit = ones(len(spo_rays[0]))*i
#        else:
#            spo_rays = hstack((spo_rays,thru_rays))
#            xou_hit = hstack((xou_hit,ones(len(thru_rays[0]))*i))
#    return spo_rays,xou_hit,clocked_rays

def SPOPetalTrace(rays):
    for i in [1]: #range(cfpar.N_xous):
        # Produces the selection of rays hitting a particular (the "ith") XOU.
        #xou_rays = SPO_ray_select(rays,cfpar.xou_loc_irs[i],cfpar.xou_loc_ors[i],cfpar.xou_widths[i],cfpar.xou_cangles[i])
        
        # Clocking the rays at the SPO clock angle.
        clocked_rays = tran.copy_rays(xou_rays)
        tran.transform(clocked_rays,0.,0.,0.,0.,0.,-cfpar.xou_cangles[i]*pi/180)
    
        # Now moving the rays to coincide with the center of of this XOU, which is a combination of its
        # radial offset and its central radius. We then tilt the module about its center by its particular
        # XOU tilt. The initial translation to the center of the XOU is done so that the tilt is performed
        # about the center of the XOU -- this better preserves the intersection of the rays with the SPO.
        r_cen = mean([cfpar.xou_loc_irs[i],cfpar.xou_loc_ors[i]])
        tran.transform(clocked_rays,0,r_cen,0,cfpar.xou_tilts[i],0,0)

        pdb.set_trace()

        # Next, we translate the rays back so that they are referenced to the optical axis of the XOU, and are traced through.
        tran.transform(clocked_rays,0,-r_cen + cfpar.xou_rad_offsets[i],0,0,0,0)
        
        # Performing the trace through the SPO under `square on' incidence conditions, where the rays may be at y slope set
        # by the cfpar.xou_tilts.
        thru_rays = SPOtrace(clocked_rays,rin = cfpar.xou_opt_irs[i],rout = cfpar.xou_opt_ors[i],azwidth = cfpar.xou_widths[i])
        
        # Undoing this transform to get to the XOU axis.
        tran.transform(thru_rays,0,0,0,-cfpar.xou_tilts[i],0,0)
        tran.transform(thru_rays,0,-cfpar.xou_rad_offsets[i],0,0,0,0)
        
        # Reversing the clock angle and tracing the rays to the focal plane.
        tran.transform(thru_rays,0.,0.,-cfpar.F,0.,0.,cfpar.xou_cangles[i]*pi/180)
        
        if i == 0:
            spo_rays = thru_rays
            xou_hit = ones(len(spo_rays[0]))*i
        else:
            spo_rays = hstack((spo_rays,thru_rays))
            xou_hit = hstack((xou_hit,ones(len(thru_rays[0]))*i))
    return spo_rays,xou_hit,clocked_rays