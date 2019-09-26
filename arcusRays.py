from numpy import *
import matplotlib.pyplot as plt
import os
import pdb
import pickle,cPickle
import copy

import PyXFocus.sources as source
import PyXFocus.surfaces as surf
import PyXFocus.analyses as anal
import PyXFocus.transformations as tran
import PyXFocus.grating as grat
import PyXFocus.conicsolve as conic

####################################################################
# Arcus Ray Class:

class ArcusRays:
    def __init__(self, PyXFocusRays, wave, order):
        self.opd = PyXFocusRays[0]
        self.x = PyXFocusRays[1]
        self.y = PyXFocusRays[2]
        self.z = PyXFocusRays[3]
        self.vx = PyXFocusRays[4]
        self.vy = PyXFocusRays[5]
        self.vz = PyXFocusRays[6]
        self.nx = PyXFocusRays[7]
        self.ny = PyXFocusRays[8]
        self.nz = PyXFocusRays[9]
        self.wave = wave
        self.order = order
        self.index = arange(len(PyXFocusRays[0]))
        self.weight = ones(len(PyXFocusRays[0]))
            
    def set_prays(self,PyXFocusRays, ind = None):
        if ind is not None:
            self.opd[ind] = PyXFocusRays[0]
            self.x[ind] = PyXFocusRays[1]
            self.y[ind] = PyXFocusRays[2]
            self.z[ind] = PyXFocusRays[3]
            self.vx[ind] = PyXFocusRays[4]
            self.vy[ind] = PyXFocusRays[5]
            self.vz[ind] = PyXFocusRays[6]
            self.nx[ind] = PyXFocusRays[7]
            self.ny[ind] = PyXFocusRays[8]
            self.nz[ind] = PyXFocusRays[9]
        else:
            self.opd = PyXFocusRays[0]
            self.x = PyXFocusRays[1]
            self.y = PyXFocusRays[2]
            self.z = PyXFocusRays[3]
            self.vx = PyXFocusRays[4]
            self.vy = PyXFocusRays[5]
            self.vz = PyXFocusRays[6]
            self.nx = PyXFocusRays[7]
            self.ny = PyXFocusRays[8]
            self.nz = PyXFocusRays[9]
            
    def yield_prays(self, ind = None):
        if ind is not None:
            return [self.opd[ind],self.x[ind],self.y[ind],self.z[ind],self.vx[ind],self.vy[ind],self.vz[ind],self.nx[ind],self.ny[ind],self.nz[ind]]
        else:
            return [self.opd,self.x,self.y,self.z,self.vx,self.vy,self.vz,self.nx,self.ny,self.nz]
    
    def yield_object_indices(self, ind):
        new_object = copy.deepcopy(self)
        for key in self.__dict__.keys():
            new_object.__dict__[key] = self.__dict__[key][ind]
        return new_object
    
    def pickle_me(self, pickle_file):
        new_object = copy.deepcopy(self)
        keys = new_object.__dict__.keys()
        attribs = []
        attribs.append(keys)
        attribs.append([new_object.__dict__[key] for key in keys])
        f = open(pickle_file,'wb')
        cPickle.dump(attribs,f)
        f.close()  

def make_channel_source(num_rays,wave,order,xextent = 500.,yextent = 675.,fs_dist = None):
    illum_height = 575
    rays = source.rectbeam(xextent/2,yextent/2,num_rays)
    rays[2] = rays[2] + illum_height
    rays[6] = -ones(len(rays[6]))
    if fs_dist is not None:
        tran.pointTo(rays,0.,illum_height,fs_dist + 12000,reverse = 1)
        zdist = surf.focusX(rays)
        rays[3] = ones(len(rays[3]))*zdist
    else:
        rays[3] = ones(len(rays[3]))*1e12
    wave = zeros(num_rays) + wave
    order = ones(num_rays)*order
    ray_object = ArcusRays(rays,wave,order)
    return ray_object

def load_ray_object_from_pickle(pickle_file):
    blank_ray_object = make_channel_source(1)
    f = open(pickle_file,'rb')
    attribs = cPickle.load(f)
    f.close()
    keys = attribs[0]
    for i in range(len(keys)):
        blank_ray_object.__dict__[keys[i]] = attribs[1][i]
    return blank_ray_object

def merge_ray_object_dict(ray_object_dict):
    try:
        merged_object = copy.deepcopy(ray_object_dict[ray_object_dict.keys()[0]])
    
    # Exception occurs when ray_object_dist is empty. In this case, return an empty ArcusRays object
    except IndexError:
        return make_channel_source(0)
    
    for ray_attrib in merged_object.__dict__.keys():
        try:
            merged_object.__dict__[ray_attrib] = hstack(tuple([ray_object_dict[key].__dict__[ray_attrib] for key in ray_object_dict]))
        except:
            pdb.set_trace()
            
    return merged_object  
