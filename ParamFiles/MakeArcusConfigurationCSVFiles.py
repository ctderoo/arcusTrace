from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb
import copy
import pickle
import string
import time
import csv

import PyXFocus.transformations as tran
import PyXFocus.surfaces as surf
import PyXFocus.sources as source

import arcusTrace.SPOPetal as ArcSPO
import arcusTrace.CATPetal as ArcCAT
import arcusTrace.DetectorArray as ArcDet
import arcusTrace.arcusUtilities as ArcUtil
import arcusTrace.arcusPerformance as ArcPerf
import arcusTrace.ParamFiles.arcus_params_rev1p8 as cfpar
import arcusTrace.arcusComponents as ArcComp
import arcusTrace.arcusRays as ArcRays

test_rays = ArcRays.make_channel_source(10**6,wave = 1240./300*10**-6)
illum_area = 45*57.

def get_facet_cyl_params(x,y):
    r = sqrt(x**2 + y**2)
    phi = arcsin(x/r)*180/pi
    return r,phi

def sort_facets_radially_azimuthally():
    stuff = asarray([(i,get_facet_cyl_params(cfpar.xgrats[i],cfpar.ygrats[i])[0],get_facet_cyl_params(cfpar.xgrats[i],cfpar.ygrats[i])[1]) for i in range(len(cfpar.xgrats))])
    rsorted_stuff = stuff[stuff[:,1].argsort()]

    for i in range(len(cfpar.row_MM1_or)):
        ind = rsorted_stuff[:,1] < cfpar.row_MM1_or[i]
        temp = rsorted_stuff[ind]
        rsorted_stuff = rsorted_stuff[~ind]
        if i == 0:
            final_sorted_stuff = temp[temp[:,2].argsort()]
        else:
            final_sorted_stuff = vstack((final_sorted_stuff,temp[temp[:,2].argsort()]))
    return final_sorted_stuff

sorted_facets = sort_facets_radially_azimuthally()
    
def facet_position_lookup(arcus_facets,arcus_xous):
    facet_pos = [(key,arcus_facets[key].facet_coords.x,arcus_facets[key].facet_coords.y) for key in arcus_facets]
    
    def xou_xy_comp(key):
        avg_r  = mean([arcus_xous[key].inner_radius,arcus_xous[key].outer_radius])
        X,Y = avg_r*sin(arcus_xous[key].clocking_angle),avg_r*cos(arcus_xous[key].clocking_angle)
        return (key,X,Y)
    
    xou_pos = [xou_xy_comp(key) for key in arcus_xous]
    
    matched_pairs = []
    for (facet_num,fx,fy) in facet_pos:
        dist = asarray([(xou_key,sqrt((xx - fx)**2 + (xy - fy)**2)) for xou_key,xx,xy in xou_pos])
        matched_pairs += [(facet_num,dist[argmin(dist[:,1].astype('float')),0])]
    
    return xou_pos,facet_pos,matched_pairs

def InitializeArcusChannel(facet_is):
    # Creating the Arcus XOUs.
    arcus_xous = dict(('OC1_XOU' + str(i),ArcComp.xou(i,float(cfpar.xou_irs[i]),float(cfpar.xou_ors[i]),float(cfpar.xou_widths[i]),float(cfpar.xou_lengths[i]),float(cfpar.xou_lengths[i]),float(cfpar.xou_cangles[i]*pi/180))) for i in range(cfpar.N_xous))
    ref_func = ArcPerf.make_reflectivity_func(cfpar.MM_coat_mat,cfpar.MM_coat_rough)
    
    for i in range(cfpar.N_xous):
        arcus_xous['OC1_XOU' + str(i)].plate_coating = 'Uncoated'
        arcus_xous['OC1_XOU' + str(i)].set_row(cfpar.arcus_row_number[i])
        arcus_xous['OC1_XOU' + str(i)].set_chan(cfpar.chan_number[i])
        arcus_xous['OC1_XOU' + str(i)].set_MM(cfpar.MM_number[i])
        arcus_xous['OC1_XOU' + str(i)].set_ref_func(ref_func)
    
    # Creating the Arcus Grating Facets.
    arcus_facets = dict(('OC1_F' + str(i),ArcComp.facet(i,cfpar.xgrats[facet_is[i]],cfpar.ygrats[facet_is[i]],cfpar.zgrats[facet_is[i]])) for i in range(len(facet_is)))
    geff_func = ArcPerf.make_geff_interp_func()
    xou_pos,facet_pos,matched_pairs = facet_position_lookup(arcus_facets,arcus_xous)
    
    for i in range(len(cfpar.xgrats)):
        arcus_facets['OC1_F' + str(i)].set_geff_func(geff_func)
    for facet,xou in matched_pairs:
        arcus_facets[facet].set_chan(1)
        arcus_facets[facet].set_SPO_MM_num(arcus_xous[xou].MM_num)
        arcus_facets[facet].set_SPO_row_num(arcus_xous[xou].row_num)
        #arcus_facets['OC1_F' + str(i)].order_select = 0
    
    # Creating the Arcus Detector Array.
    num2alpha = dict(zip(range(0, 26), string.ascii_uppercase))
    #arcus_detectors = dict(('CCD_' + num2alpha[i],ArcComp.ccd_detector(i,2058,1039,\
    #                    cfpar.det_locs[i][0],cfpar.det_locs[i][1],cfpar.det_locs[i][2],\
    #                    cfpar.det_vecs[i][0],cfpar.det_vecs[i][1],cfpar.det_vecs[i][2])) for i in range(len(cfpar.det_locs[:,0])))
    arcus_detectors = dict(('CCD_' + num2alpha[i],ArcComp.ccd_detector(i,2058,1039,\
                    cfpar.instrum_det_locs[i][0],cfpar.instrum_det_locs[i][1],cfpar.instrum_det_locs[i][2],\
                    cfpar.instrum_det_vecs[i][0],cfpar.instrum_det_vecs[i][1],cfpar.instrum_det_vecs[i][2])) for i in range(len(cfpar.instrum_det_locs[:,0])))
    for i in range(len(cfpar.instrum_det_locs[:,0])):
        arcus_detectors['CCD_' + num2alpha[i]].add_det_effect('QE',ArcPerf.det_qe_interp_func)
        arcus_detectors['CCD_' + num2alpha[i]].add_det_effect('Contamination',ArcPerf.det_contam_interp_func)
        arcus_detectors['CCD_' + num2alpha[i]].add_det_effect('Optical Blocking',ArcPerf.det_optblock_interp_func)
        arcus_detectors['CCD_' + num2alpha[i]].add_det_effect('UV Blocking',ArcPerf.det_uvblock_interp_func)
        arcus_detectors['CCD_' + num2alpha[i]].add_det_effect('Mesh Blocking',ArcPerf.det_simesh_interp_func)
    
    return arcus_xous,arcus_facets,arcus_detectors

times = []
times.append(time.time())
arcus_xous,arcus_facets,arcus_detectors = InitializeArcusChannel(list(sorted_facets[:,0].astype(int)))
times.append(time.time())




########
#spo_petal_rays = ArcSPO.SPOPetalTrace(test_rays,arcus_xous)
#times.append(time.time())
#
#cat_petal_rays = ArcCAT.CATPetalTrace(spo_petal_rays,arcus_facets)
#times.append(time.time())
#
#det_rays = ArcDet.DetectorArrayTrace(cat_petal_rays,arcus_detectors)
#times.append(time.time())
#
#durations = [times[i + 1] - times[i] for i in range(len(times) - 1)]


def write_spos_to_bitbucket(arcus_xous,csv_fn = 'Arcus_SPO_XOU_Specs_Rev1p0_171112.csv'):
    sorted_xous = zip(*sorted(tuple([(arcus_xous[key].xou_num,arcus_xous[key]) for key in arcus_xous.keys()])))
    
    keys_to_track = ['xou_num','chan_id','MM_num','row_num','inner_radius','outer_radius',\
                     'azwidth','clocking_angle','primary_length','secondary_length',\
                     'focal_length','pore_space','plate_height','plate_coating']
    keys_units = ['-','-','-','-','mm','mm','mm','radians','mm','mm','mm','mm','mm','-']
    keys_datatypes = ['int','int','int','int','float64','float64','float64','float64',\
                      'float64','float64','float64','float64','float64','string']
    
    
    with open(csv_fn,'wb') as csvfile:
        csvfile.write('# %ECSV 0.9\n')
        csvfile.write('# ---\n')
        csvfile.write('# datatype:\n')
        
        for m in range(len(keys_to_track)):
            string_to_write = '# - {name: ' + keys_to_track[m]
            if keys_units[m] is not '-':
                string_to_write += ',' + 'unit: ' + keys_units[m]
            if keys_datatypes[m] is not '-':
                string_to_write += ',' + 'datatype: ' + keys_datatypes[m]  
            string_to_write += '}\n'
            csvfile.write(string_to_write)
        
        csvfile.write('# Author: Casey DeRoo\n')
        csvfile.write('# ORIGFILE: arcusTrace.ParamFiles.arcus_params_rev1p8.py\n')
        csvfile.write('# Description: Specifications of SPOs from ARCUS-MEMO-#0002 (E. Hertz, as of 17/11/12)' + '\n')
        
        writer = csv.writer(csvfile,delimiter = ',')
        writer.writerow(keys_to_track)
        for i in range(len(sorted_xous[0])):
            line_to_write = [sorted_xous[1][i].__dict__[key] for key in keys_to_track]
            writer.writerow(line_to_write)
    
    csvfile.close()
    return

def write_facets_to_bitbucket(arcus_facets,csv_fn = 'Arcus_CATGrating_Facets_Specs_Rev1p0_171112.csv'):
    sorted_facets = zip(*sorted(tuple([(arcus_facets[key].facet_num,arcus_facets[key]) for key in arcus_facets.keys()])))
    
    column_names = ['facet_num','chan_id','SPO_MM_num','SPO_row_num',\
                     'X','Y','Z',\
                     'DispNX','DispNY','DispNZ',\
                     'GBarNX','GBarNY','GBarNZ',\
                     'NormNX','NormNY','NormNZ',\
                     'xsize','ysize','period']
    keys_to_track = [['facet_num'],['chan_id'],['SPO_MM_num'],['SPO_row_num'],\
                     [['facet_coords'],['x']],[['facet_coords'],['y']],[['facet_coords'],['z']],\
                     [['facet_coords'],['xhat'],[0]],[['facet_coords'],['xhat'],[1]],[['facet_coords'],['xhat'],[2]],\
                     [['facet_coords'],['yhat'],[0]],[['facet_coords'],['yhat'],[1]],[['facet_coords'],['yhat'],[2]],\
                     [['facet_coords'],['zhat'],[0]],[['facet_coords'],['zhat'],[1]],[['facet_coords'],['zhat'],[2]],\
                     ['xsize'],['ysize'],['period']]
    column_units = ['-','-','-','-','mm','mm','mm','-','-','-','-','-','-','-','-','-','mm','mm','mm']
                
    column_datatypes = ['int','int','int','int'] + ['float64']*15
    
    with open(csv_fn,'wb') as csvfile:
        csvfile.write('# %ECSV 0.9\n')
        csvfile.write('# ---\n')
        csvfile.write('# datatype:\n')
        
        for m in range(len(keys_to_track)):
            string_to_write = '# - {name: ' + column_names[m]
            if column_units[m] is not '-':
                string_to_write += ',' + 'unit: ' + column_units[m]
            if column_datatypes[m] is not '-':
                string_to_write += ',' + 'datatype: ' + column_datatypes[m]  
            string_to_write += '}\n'
            csvfile.write(string_to_write)
        
        csvfile.write('# Author: Casey DeRoo\n')
        csvfile.write('# ORIGFILE: arcusTrace.ParamFiles.arcus_params_rev1p8.py\n')
        csvfile.write('# Description: Specifications of CAT Grating Facets from ARCUS-MEMO-#0002 (E. Hertz, as of 17/11/12)' + '\n')
        
        writer = csv.writer(csvfile,delimiter = ',')
        writer.writerow(column_names)
        for i in range(len(sorted_facets[0])):
            line_to_write = []
            for key in keys_to_track:
                if len(key) == 1:
                    line_to_write += [sorted_facets[1][i].__dict__[key[0]]]
                elif len(key) == 2:
                    line_to_write += [sorted_facets[1][i].__dict__[key[0][0]].__dict__[key[1][0]]]
                elif len(key) == 3:
                    line_to_write += [sorted_facets[1][i].__dict__[key[0][0]].__dict__[key[1][0]][key[2]][0]]
                else:
                    print 'What the fuck'
                    pdb.set_trace()
            writer.writerow(line_to_write)
    
    csvfile.close()
    return

def write_detectors_to_bitbucket(arcus_detectors,csv_fn = 'Arcus_DetectorArray_Specs_Rev1p0_171112.csv'):
    sorted_dets = zip(*sorted(tuple([(arcus_detectors[key].ccd_num,arcus_detectors[key]) for key in arcus_detectors.keys()])))
    
    column_names = ['ccd_num',\
                     'X','Y','Z',\
                     'XHat_NX','XHat_NY','XHat_NZ',\
                     'YHat_NX','YHat_NY','YHat_NZ',\
                     'ZHat_NX','ZHat_NY','ZHat_NZ',\
                     'xwidth','ywidth','xpix','ypix','xpixsize','ypixsize']
    keys_to_track = [['ccd_num'],\
                     [['ccd_coords'],['x']],[['ccd_coords'],['y']],[['ccd_coords'],['z']],\
                     [['ccd_coords'],['xhat'],[0]],[['ccd_coords'],['xhat'],[1]],[['ccd_coords'],['xhat'],[2]],\
                     [['ccd_coords'],['yhat'],[0]],[['ccd_coords'],['yhat'],[1]],[['ccd_coords'],['yhat'],[2]],\
                     [['ccd_coords'],['zhat'],[0]],[['ccd_coords'],['zhat'],[1]],[['ccd_coords'],['zhat'],[2]],\
                     ['xwidth'],['ywidth'],['xpix'],['ypix'],['xpixsize'],['ypixsize']]
    column_units = ['-','mm','mm','mm','-','-','-','-','-','-','-','-','-','mm','mm','-','-','mm','mm']
                
    column_datatypes = ['int'] + ['float64']*18
    
    pdb.set_trace()
    with open(csv_fn,'wb') as csvfile:
        csvfile.write('# %ECSV 0.9\n')
        csvfile.write('# ---\n')
        csvfile.write('# datatype:\n')
        
        for m in range(len(keys_to_track)):
            string_to_write = '# - {name: ' + column_names[m]
            if column_units[m] is not '-':
                string_to_write += ',' + 'unit: ' + column_units[m]
            if column_datatypes[m] is not '-':
                string_to_write += ',' + 'datatype: ' + column_datatypes[m]  
            string_to_write += '}\n'
            csvfile.write(string_to_write)
        
        csvfile.write('# Author: Casey DeRoo\n')
        csvfile.write('# ORIGFILE: arcusTrace.ParamFiles.arcus_params_rev1p8.py\n')
        csvfile.write('# Description: Specifications of Arcus Focal Plane, Referenced to Global Instrument Coordinate System' + '\n')
        
        writer = csv.writer(csvfile,delimiter = ',')
        writer.writerow(column_names)
        for i in range(len(sorted_dets[0])):
            line_to_write = []
            for key in keys_to_track:
                if len(key) == 1:
                    line_to_write += [sorted_dets[1][i].__dict__[key[0]]]
                elif len(key) == 2:
                    line_to_write += [sorted_dets[1][i].__dict__[key[0][0]].__dict__[key[1][0]]]
                elif len(key) == 3:
                    line_to_write += [sorted_dets[1][i].__dict__[key[0][0]].__dict__[key[1][0]][key[2]][0]]
                else:
                    print 'What the fuck'
                    pdb.set_trace()
            writer.writerow(line_to_write)
    
    csvfile.close()
    return


#write_spos_to_bitbucket(arcus_xous)
#write_facets_to_bitbucket(arcus_facets)
#write_detectors_to_bitbucket(arcus_detectors)