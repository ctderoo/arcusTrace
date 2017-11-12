from numpy import *
import PyXFocus.transformations as tran
import matplotlib.pyplot as plt

import pdb
from scipy.optimize import root

############################################

class coordinate_system:
    def __init__(self,x,y,z,xhat,yhat,zhat):
        self.x = x
        self.y = y
        self.z = z
        self.xhat = xhat
        self.yhat = yhat
        self.zhat = zhat

instrument_coord_sys = coordinate_system(0.,0.,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))

############################################

class xou:
    def __init__(self, xou_num, inner_radius, outer_radius, azwidth, plength, slength, clock_ang):
        # Tracking which XOU number this is.
        self.xou_num = xou_num
        
        # Geometric parameters
        self.focal_length = 12000.
        self.pore_space = 0.605
        self.plate_height = 0.775
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.plate_radii = arange(self.inner_radius,self.outer_radius,self.plate_height)
        self.azwidth = azwidth
        self.sp_space = 0.100
        self.primary_length = plength
        self.secondary_length = slength
        
        # XOU-specific coordinate system, as specified from the instrument coordinate system.
        self.xou_coords = coordinate_system(0.,0.,12000.,array([cos(clock_ang),-sin(clock_ang),0.]),array([sin(clock_ang),cos(clock_ang),0.,]),array([0.,0.,1.,]))
        
        # SPO raytracing specific parameters
        self.scatter = True
        self.dispdir_scatter_val = 1.93/2.35*5e-6
        self.crossdispdir_scatter_val = 15.5/2.35*5e-6
        self.ref_func = None
        self.plate_coating = 'NA'
        self.plate_roughness = NaN
        self.pore_vignette = True
        
    def set_ref_func(self, ref_func):
        self.ref_func = ref_func
    
    def set_row(self,row_num):
        self.row_num = row_num
        
    def set_MM(self, MM_num):
        self.MM_num = MM_num
    
    def set_chan(self, chan_id):
        self.chan_id = chan_id
        
############################################

class facet(object):
    def __init__(self,facet_num,xloc,yloc,zloc):
        # Setting the physical characteristics of the grating facet.
        self.facet_num = facet_num
        self.xsize = 27
        self.ysize = 26
        self.period = 2.00e-4
        
        self.alpha = 1.8*pi/180
        self.xi = 2.892138*pi/180
        self.r = 5945.787
        self.R = 5915.513
        
        # Actually orienting the grating facets based on the input parameters.
        gdisp,gbar,ngrat = self.__compute_facet_orientation(xloc,yloc,zloc)
        self.facet_coords = coordinate_system(xloc,yloc,zloc,gdisp,gbar,ngrat)

        # Grating facet -- raytracing specific parameters
        self.order_select = None
        self.L1supp = True
        self.L2supp = True
        self.debye_waller = True

    def set_geff_func(self, geff_func):
        self.geff_func = geff_func

    def set_chan(self, chan_id):
        self.chan_id = chan_id
    
    def set_window(self,window_num):
        self.window_num = window_num
    
    def compute_facet_orientation(self,xloc,yloc,zloc):
        # First, pointing the gratings towards the telescope focus -- setting normal incidence while being positioned on the Rowland torus.
        # The nominal normal of the transmission gratings, focus_ngrat, points towards the focus of telescope.
        norm_ind = array([xloc,yloc,zloc])
        focus_ngrat = norm_ind/linalg.norm(norm_ind)
        
        # Constructing a grating bar vector that's 1) orthogonal to the normal vector and 2) orthogonal to x_hat. The
        # weighting done in the third slot ensures orthogonality.
        gbar = array([0,1.,-focus_ngrat[1]/focus_ngrat[2]])
        gbar = gbar/linalg.norm(gbar)
        
        # Now rotating about the grating bar to set the desired blaze angle. First, we produce a rotation matrix to
        # rotate another vector about the grating bar direction by the desired incidence angle alpha. Then we rotate
        # the nominal transmission grating normal (pointed towards the focus) about the grating bar direction by the
        # incidence angle.
        ngrat = dot(tran.tr.rotation_matrix(self.alpha,gbar)[:3,:3],focus_ngrat)

        # Finally, constructing the orthogonal vector for each grating representing the local dispersion direction.
        gdisp = cross(gbar,ngrat)

        #tgrat,pgrat = gdisp,gbar
        return gdisp,gbar,ngrat
    
    __compute_facet_orientation = compute_facet_orientation

############################################

class det_filter(object):
    def __init__(self,filter_name,filter_func):
        self.filter_name = filter_name
        self.filter_func = filter_func

class ccd_detector(object):
    def __init__(self,ccd_num,xpix,ypix,xcent,ycent,zcent,xhat,yhat,zhat):
        '''
        xhat and zhat are the dispersion direction and the CCD normal respectively.
        '''
        self.ccd_num = ccd_num
        
        self.xpixsize = 0.024
        self.ypixsize = 0.024
        self.xpix = xpix
        self.ypix = ypix
        self.xwidth = self.xpix*self.xpixsize
        self.ywidth = self.ypix*self.ypixsize
        
        self.ccd_coords = coordinate_system(xcent,ycent,zcent,xhat,yhat,zhat)
        self.det_effects = []
        
    def add_det_effect(self,filter_name,filter_fn):
        self.det_effects.append(det_filter(filter_name,filter_fn))

#
#det_xstart = 540
#det_xstep = 2048*0.024 + 0.5
#det_xlocs = -(det_xstart + arange(8)*det_xstep)
#det_RoC = mean(zgrats)*2
#
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
#
#det_locs,det_vecs = define_det_array(det_xlocs,det_RoC)