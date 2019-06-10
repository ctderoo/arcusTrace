from numpy import *
import PyXFocus.transformations as tran
import matplotlib.pyplot as plt
import string

import pdb
from scipy.optimize import root

import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.arcusUtilities as ArcUtil
from arcusTrace.ParamFiles.pointers import *

############################################

class coordinate_system:
    def __init__(self,x,y,z,xhat,yhat,zhat):
        self.x = x
        self.y = y
        self.z = z
        self.xhat = xhat
        self.yhat = yhat
        self.zhat = zhat
    
    def unpack(self):
        return self.x,self.y,self.z,self.xhat,self.yhat,self.zhat
    
instrument_coord_sys = coordinate_system(0.,0.,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))
OC1_coords = coordinate_system(300.,2.5,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))
OC2_coords = coordinate_system(300.,-7.5,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))
OC3_coords = coordinate_system(-300.,-2.5,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))
OC4_coords = coordinate_system(-300.,7.5,0.,array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))

#OC2_coords = coordinate_system(300.,-7.5,0.,array([1.,0.,0.]),array([0.,-1.,0.]),array([0.,0.,1.]))
#OC3_coords = coordinate_system(-300.,-2.5,0.,array([-1.,0.,0.]),array([0.,-1.,0.]),array([0.,0.,1.]))
#OC4_coords = coordinate_system(-300.,7.5,0.,array([-1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.]))

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
        self.clocking_angle = clock_ang
        
        # XOU-specific coordinate system, as specified from the instrument coordinate system.
        self.xou_coords = coordinate_system(0.,0.,12000.,array([cos(clock_ang),-sin(clock_ang),0.]),array([sin(clock_ang),cos(clock_ang),0.,]),array([0.,0.,1.,]))
        
        # SPO raytracing specific parameters
        self.scatter = ['Gaussian','Gaussian']
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
        self.grat_scatter_val = None
        
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
        
    def set_SPO_MM_num(self, SPO_MM_num):
        self.SPO_MM_num = SPO_MM_num
    
    def set_SPO_row_num(self,SPO_row_num):
        self.SPO_row_num = SPO_row_num
    
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
        xhat and zhat are the dispersion direction and the CCD normal respectively. Note that the
        coordinate system is given for the corner of the CCD.
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

############################################
# Entire channel class here.
#
def read_caldb_csvfile(fn):
    '''
    Read in the .csv files from the CALDB format.
    Input:
    fn -- filename to be read in
    Output:
    header -- the column labels as strings
    data -- the contents of the csvfile as a float.
    '''
    file_contents = genfromtxt(fn,comments = '#',dtype = str)
    header,data = file_contents[0],file_contents[1:]
    return header,data

class ArcusChannel(object):
    ''' Default class variables '''
    # default_xou_pointer = 'C:/Users/swarm/Software/python_repository/arcusTrace/ParamFiles/Arcus_SPO_XOU_Specs_Rev1p0_171112.csv'
    # default_xou_ref_pointer = 'C:/Users/swarm/Software/ReflectLib/SL_LSi02_LThick1p0_LRough0p4_SSi_SRough0p0_Pol1_CXRO.npy'
    #MLMirror_L1SiC_L2Ir_SubSi_L1Thick_6p0L2Thick_10p0_Rough4p00AngRMS_X-rayRefData.npy'
    # default_facet_pointer = 'C:/Users/swarm/Software/python_repository/arcusTrace/ParamFiles/Arcus_CATGrating_Facets_Specs_Rev1p0_171112.csv'
    # default_diff_eff_pointer = 'C:/Users/swarm/Software/Bitbucket/caldb-inputdata/gratings/Si_4um_deep_30pct_dc_extended.csv'
    
    def __init__(self,chan_num, chan_coords = OC1_coords,xou_pointer = default_xou_pointer,facet_pointer = default_facet_pointer,\
               xou_ref_pointer = default_xou_ref_pointer,order_select = None, diff_eff_pointer = default_diff_eff_pointer):

        self.chan_num = chan_num
        self.xou_pointer = xou_pointer
        self.facet_pointer = facet_pointer
        self.xou_ref_pointer = xou_ref_pointer
        self.order_select = None
        self.diff_eff_pointer = diff_eff_pointer
        
        self.set_reflections(chan_num)
        self.chan_xous = self.set_chan_xous(xou_pointer,xou_ref_pointer)
        self.chan_facets = self.set_chan_facets(facet_pointer,order_select,diff_eff_pointer)
        self.chan_coords = chan_coords
    
    def set_reflections(self,chan_num):
        self.refx = False
        self.refy = False
        
        if logical_or(chan_num == 3,chan_num == 4):
            self.refx = True
        if logical_or(chan_num == 2,chan_num == 3):
            self.refy = True
            
    def set_chan_xous(self,xou_pointer,xou_ref_pointer):
        xou_header,xou_data = read_caldb_csvfile(xou_pointer)
        if xou_ref_pointer is not None:
            ref_func = ArcPerf.make_reflectivity_func(xou_ref_pointer)
        xou_header = xou_header.split(",")
        
        # Keys needed to initialize the Arcus XOUs.
        needed_init_keys = ['xou_num','chan_id','MM_num','row_num','inner_radius','outer_radius','azwidth','clocking_angle','primary_length','secondary_length']
        dtype_init_keys = [int,int,int,int,float,float,float,float,float,float]
        def selector(values,ind,dtypes = dtype_init_keys):
            return dtypes[ind](values[ind])
        
        ind = [xou_header.index(needed_init_keys[i]) for i in range(len(needed_init_keys))]
        
        chan_xous = dict()
        for i in range(len(xou_data)):
            xou_chars = xou_data[i].split(",")
            xou_num,chan_id,MM_num,row_num,inner_radius,outer_radius,azwidth,clock_ang,plength,slength = [selector(xou_chars,ind[j]) for j in range(len(ind))]
            chan_xous['XOU' + str(xou_num)] = xou(xou_num,inner_radius,outer_radius,azwidth,plength,slength,clock_ang)
            chan_xous['XOU' + str(xou_num)].set_row(row_num)
            chan_xous['XOU' + str(xou_num)].set_chan(chan_id)
            chan_xous['XOU' + str(i)].set_MM(MM_num)
            if xou_ref_pointer is not None:
                chan_xous['XOU' + str(i)].set_ref_func(ref_func)
        
        return chan_xous
    
    def set_chan_facets(self,facet_pointer,order_select,diff_eff_pointer):
        facet_header,facet_data = read_caldb_csvfile(facet_pointer)
        if order_select is None:
            geff_func = ArcPerf.make_geff_interp_func(diff_eff_pointer,style = 'New')
        facet_header = facet_header.split(",")
        
        # Keys needed to initialize the Arcus XOUs.
        needed_init_keys = ['facet_num','chan_id','SPO_MM_num','SPO_row_num','X','Y','Z',\
                            'DispNX','DispNY','DispNZ','GBarNX','GBarNY','GBarNZ','NormNX','NormNY','NormNZ','xsize','ysize','period']
        dtype_init_keys = [int,int,int,int,float,float,float,\
                           float,float,float,float,float,float,float,float,float,float,float,float]
        
        def selector(values,ind,dtypes = dtype_init_keys):
            return dtypes[ind](values[ind])
        
        ind = [facet_header.index(needed_init_keys[i]) for i in range(len(needed_init_keys))]
        
        chan_facets = dict()
        for i in range(len(facet_data)):
            facet_chars = facet_data[i].split(",")
            facet_num,chan_id,SPO_MM_num,SPO_row_num,X,Y,Z,\
                            DispNX,DispNY,DispNZ,GBarNX,GBarNY,GBarNZ,NormNX,NormNY,NormNZ,xsize,ysize,period= [selector(facet_chars,ind[j]) for j in range(len(ind))]
            facet_coords_system = coordinate_system(X,Y,Z,array([DispNX,DispNY,DispNZ]),array([GBarNX,GBarNY,GBarNZ]),array([NormNX,NormNY,NormNZ]))
            
            chan_facets['F' + str(facet_num)] = facet(facet_num,X,Y,Z)
            chan_facets['F' + str(facet_num)].facet_coords = facet_coords_system
            
            chan_facets['F' + str(facet_num)].set_chan(chan_id)
            chan_facets['F' + str(facet_num)].set_SPO_MM_num(SPO_MM_num)
            chan_facets['F' + str(facet_num)].set_SPO_row_num(SPO_row_num)
            chan_facets['F' + str(facet_num)].order_select = order_select
            if order_select is None:
                chan_facets['F' + str(facet_num)].set_geff_func(geff_func)
        
        return chan_facets
    
    def rays_from_chan_to_instrum_coords(self, ray_object,inverse = False):
        prays = ray_object.yield_prays()
        moved_prays = ArcUtil.chan_to_instrum_transform(prays,self.chan_coords,self.refx,self.refy,inverse = inverse)
        ray_object.set_prays(moved_prays)    
    
class ArcusFPA(object):
    ''' Class wide variables go here'''
    # default_det_pointer = 'C:/Users/swarm/Software/python_repository/arcusTrace/ParamFiles/Arcus_DetectorArray_Specs_Rev3p0_171211.csv'
    # bitbucket_path = 'C:/Users/swarm/Software/Bitbucket/caldb-inputdata'
    default_det_qe_fn = bitbucket_path + '/detectors/qe.csv'
    default_det_contam_fn = bitbucket_path + '/detectors/contam.csv'
    default_opt_block_fn = bitbucket_path + '/filters/opticalblocking.csv'
    default_uv_block_fn = bitbucket_path + '/filters/sifilter.csv'
    default_Si_mesh_block_fn = bitbucket_path + '/filters/uvblocking.csv'

    def __init__(self,fpa_coords = instrument_coord_sys,det_pointer = default_det_pointer,
                 det_qe_fn = default_det_qe_fn,det_contam_fn = default_det_contam_fn,opt_block_fn = default_opt_block_fn, uv_block_fn = default_uv_block_fn,Si_mesh_block_fn = default_Si_mesh_block_fn):
        '''
        Turn any of the paths to None to turn off these effects.
        '''
        self.fpa_dets = self.set_dets(det_pointer)
        self.fpa_coords = instrument_coord_sys

        if det_qe_fn is not None:
            det_qe_interp_func = ArcPerf.make_detector_effect_func(det_qe_fn)
            self.set_det_effect(det_qe_interp_func,'QE')
        if det_contam_fn is not None:
            det_contam_interp_func = ArcPerf.make_detector_effect_func(det_contam_fn)
            self.set_det_effect(det_contam_interp_func,'Contamination')
        if opt_block_fn is not None:
            det_optblock_interp_func = ArcPerf.make_detector_effect_func(opt_block_fn)
            self.set_det_effect(det_optblock_interp_func,'Optical Blocking')
        if uv_block_fn is not None:
            det_uvblock_interp_func = ArcPerf.make_detector_effect_func(uv_block_fn)
            self.set_det_effect(det_uvblock_interp_func,'UV Blocking')
        if Si_mesh_block_fn is not None:
            det_simesh_interp_func = ArcPerf.make_detector_effect_func(Si_mesh_block_fn)
            self.set_det_effect(det_simesh_interp_func,'Mesh Blocking')
            
    def set_dets(self,det_pointer):
        det_header,det_data = read_caldb_csvfile(det_pointer)
        det_header = det_header.split(",")
        
        # Keys needed to initialize the Arcus XOUs.
        needed_init_keys = ['ccd_num','X','Y','Z',\
                            'XHat_NX','XHat_NY','XHat_NZ','YHat_NX','YHat_NY','YHat_NZ','ZHat_NX','ZHat_NY','ZHat_NZ','xwidth','ywidth','xpix','ypix','xpixsize','ypixsize']
        dtype_init_keys = [int,float,float,float,\
                           float,float,float,float,float,float,float,float,float,float,float,int,int,float,float]
        
        def selector(values,ind,dtypes = dtype_init_keys):
            return dtypes[ind](values[ind])
        
        ind = [det_header.index(needed_init_keys[i]) for i in range(len(needed_init_keys))]
        
        fpa_ccds = dict()
        num2alpha = dict(zip(range(0, 26), string.ascii_uppercase))
        
        for i in range(len(det_data)):
            det_chars = det_data[i].split(",")
            
            ccd_num,X,Y,Z,\
            XHat_NX,XHat_NY,XHat_NZ,YHat_NX,YHat_NY,YHat_NZ,ZHat_NX,ZHat_NY,ZHat_NZ,\
            xwidth,ywidth,xpix,ypix,xpixsize,ypixsize = [selector(det_chars,ind[j]) for j in range(len(ind))]
            
            fpa_ccds['CCD_' + num2alpha[i]] = ccd_detector(ccd_num,xpix,ypix,X,Y,Z,array([XHat_NX,XHat_NY,XHat_NZ]),array([YHat_NX,YHat_NY,YHat_NZ]),array([ZHat_NX,ZHat_NY,ZHat_NZ]))
        return fpa_ccds

    def set_det_effect(self,det_effect_func, det_effect_name = 'N/A'):
        for key in self.fpa_dets.keys():
            self.fpa_dets[key].add_det_effect(det_effect_name,det_effect_func)
            
    def update_fpa_coords(self, new_fpa_coords):
        fpa_x,fpa_y,fpa_z,fpa_xhat,fpa_yhat,fpa_zhat = new_fpa_coords.unpack()
        for key in self.fpa_dets.keys():
            x,y,z,xhat,yhat,zhat = self.fpa_dets[key].ccd_coords.unpack()
            
            rot_fpa = linalg.inv(ArcUtil.make_rot_matrix(fpa_xhat,fpa_yhat,fpa_zhat))
            new_xhat,new_yhat,new_zhat = dot(rot_fpa,xhat),dot(rot_fpa,yhat),dot(rot_fpa,zhat)
            self.fpa_dets[key].ccd_coords = coordinate_system(x + fpa_x,y + fpa_y,z + fpa_z,new_xhat,new_yhat,new_zhat)
            
    
    
