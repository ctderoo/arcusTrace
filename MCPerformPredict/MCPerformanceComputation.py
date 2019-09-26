from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb
import copy
import cPickle
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
import arcusTrace.arcusComponents as ArcComp
import arcusTrace.arcusRays as ArcRays

import arcusTrace.MCPerformPredict.MCPerformancePlotting as ArcMCPlot

################################################
# Functions for Performance Computation
################################################

illum_area = 45*57.

def single_channel_trace(opt_chan,det_array,wavelength,order,N,fs_dist = None):
    # Create the source, trace it through the SPO petal and the CAT grating petal.
    test_rays = ArcRays.make_channel_source(N,order = order,wave = wavelength,fs_dist = fs_dist)
    spo_petal_rays = ArcSPO.SPOPetalTrace(test_rays,opt_chan.chan_xous)

    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
    if len(spo_petal_rays.x) == 0:
        return spo_petal_rays

    cat_petal_rays = ArcCAT.CATPetalTrace(spo_petal_rays,opt_chan.chan_facets)

    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
    if len(cat_petal_rays.x) == 0:
        return cat_petal_rays

    # Do transform to petal coordinates.
    instrum_chan_rays = copy.deepcopy(cat_petal_rays)
    opt_chan.rays_from_chan_to_instrum_coords(instrum_chan_rays)
    instrum_chan_rays.chan_num = ones(len(instrum_chan_rays.x))*opt_chan.chan_num
    
    if len(instrum_chan_rays.order) != len(instrum_chan_rays.x):
        pdb.set_trace()

    # Hit the shared Arcus Focal Plane.
    det_rays = ArcDet.DetectorArrayTrace(instrum_chan_rays,det_array.fpa_dets)
    
    print len(instrum_chan_rays.x) - len(det_rays.x)

    if len(det_rays.order) != len(det_rays.x):
        pdb.set_trace()

    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
    if len(det_rays.x) == 0:
        return det_rays

    # Transform back to petal coordinates for performance evaluation.
    opt_chan.rays_from_chan_to_instrum_coords(det_rays,inverse = True)
    
    return det_rays

def compute_order_res(chan_rays,order,threshold = 0):
    if order == 0:
        return 0
    else:
        try: 
            x,FWHMx = abs(mean(chan_rays.x[order_ind])),ArcUtil.compute_FWHM(chan_rays.x[order_ind])
        except:
            pdb.set_trace()
        return x/FWHMx

def compute_order_EA(chan_rays,order,convert_factor):
    #order_ind = chan_rays.order == order
    return sum(chan_rays.weight)*convert_factor

#def compute_perf_by_orders(chan_rays,convert_factor,all_orders = range(0,13,1)):
#    res = asarray([compute_order_res(chan_rays,order) for order in all_orders])
#    ea = asarray([compute_order_EA(chan_rays,order,convert_factor) for order in all_orders])
#    return res,ea

def ArcusMCTrace(opt_chans,fpa,wavelengths,N,orders,fileend,pickle_path):    
    # Scanning across optical channel (i), wavelength (j) , and order (k).
    ArcusR,ArcusEA = zeros((len(opt_chans),len(wavelengths),len(orders))),zeros((len(opt_chans),len(wavelengths),len(orders)))

    instrum_ray_dict = {}
    for i in range(len(opt_chans)):
        for j in range(len(wavelengths)):
            for k in range(len(orders)):
                print 'Computing ray bundle -- energy: ' + "{:5.1f}".format(1240./(wavelengths[j]*10**6)) + ' eV, wavelength: ' + "{:1.3f}".format(wavelengths[j]*10**6) + ' nm, order: ' + str(orders[k]) + ', OC' + str(i + 1)
                chan_rays = single_channel_trace(opt_chans[i],fpa,wavelengths[j],orders[k],N)
   
                # Checking if the ray dictionary is empty. If it is, if not chan_rays returns True and we scrap that measurement.
                if len(chan_rays.x) == 0:
                    ArcusR[i,j,k],ArcusEA[i,j,k] = 0,0
                else:
                    ArcusR[i,j,k],ArcusEA[i,j,k] = compute_order_res(chan_rays,orders[k]),compute_order_EA(chan_rays,orders[k],convert_factor = illum_area/float(N))
                    
                    compute_perf_by_orders(chan_rays,convert_factor = illum_area/float(N),all_orders = orders)
                    # Converting the rays back to instrument coordinates for storage.
                    try:
                        opt_chans[i].rays_from_chan_to_instrum_coords(chan_rays)
                    except:
                        pdb.set_trace()
                    instrum_ray_dict['OC' + str(i) + '_WaveStep' + str(j)] = chan_rays
    
            if j % 20 == 0:
                dict_for_merge = copy.deepcopy(instrum_ray_dict)
                instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
                f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
                print '\n' + 'Dumping to pickle file....' + '\n'
                cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
                instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1'))
                f.close()
   
    dict_for_merge = copy.deepcopy(instrum_ray_dict)
    instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
    print '\n' + 'Dumping to pickle file....' + '\n'
    f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
    cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
    f.close()
    instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1'))
    return ArcusR,ArcusEA

#def ArcusMCTrace(opt_chans,fpa,wavelengths,N,orders,fileend,pickle_path):    
#    # Scanning across optical channel (i), wavelength (j) , and order (k).
#    ArcusR,ArcusEA = zeros((len(opt_chans),len(wavelengths),len(orders))),zeros((len(opt_chans),len(wavelengths),len(orders)))

#    instrum_ray_dict = {}
#    for i in range(len(opt_chans)):
#        for j in range(len(wavelengths)):
#            print 'Computing ray bundle -- energy: ' + "{:5.1f}".format(1240./(wavelengths[j]*10**6)) + ' eV, wavelength: ' + "{:1.3f}".format(wavelengths[j]*10**6) + ' nm, OC' + str(i + 1)
#            chan_rays = single_channel_trace(opt_chans[i],fpa,wavelengths[j],N)
            
#            # Checking if the ray dictionary is empty. If it is, if not chan_rays returns True and we scrap that measurement.
#            if len(chan_rays.x) == 0:
#                ArcusR[i,j],ArcusEA[i,j] = 0,0
#                #instrum_ray_dict['OC' + str(i) + '_WaveStep' + str(j)] = chan_rays
#            else:
#                ArcusR[i,j],ArcusEA[i,j] = compute_perf_by_orders(chan_rays,convert_factor = illum_area/float(N),all_orders = orders)
#                # Converting the rays back to instrument coordinates for storage.
#                try:
#                    opt_chans[i].rays_from_chan_to_instrum_coords(chan_rays)
#                except:
#                    pdb.set_trace()
#                instrum_ray_dict['OC' + str(i) + '_WaveStep' + str(j)] = chan_rays
    
#            if j % 20 == 0:
#                dict_for_merge = copy.deepcopy(instrum_ray_dict)
#                instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
#                f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
#                print '\n' + 'Dumping to pickle file....' + '\n'
#                cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#                instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1'))
#                f.close()
   
#    dict_for_merge = copy.deepcopy(instrum_ray_dict)
#    instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
#    print '\n' + 'Dumping to pickle file....' + '\n'
#    f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
#    cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#    f.close()
#    instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1'))
#    return ArcusR,ArcusEA

#def do_rowbyrow_recalc(wavelengths,orders,N,fileend,pickle_path):
#    # Loading the rays from the _Rays.pk1 pickle file.
#    print 'Loading rays from pickle file....'
#    path = pickle_path + 'Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1')
#    ray_object = ArcRays.load_ray_object_from_pickle(path)
#    print 'Loaded!'

#    ###################################
#    # Establishing functions for row mapping.
#    # Row number by XOU.
#    row_number = array([1,1,1,1,1,1,1,1,
#                        2,2,2,2,2,2,2,2,
#                        3,3,3,3,3,3,3,3,3,3,3,3,
#                        4,4,4,4,4,4,4,4,
#                        5,5,5,5,5,5,5,5,
#                        6,6,6,6,6,6,6,6,
#                        7,7,7,7,7,7,7,7,
#                        8,8,8,8,8,8,8,8])
#    xou_by_row = tuple(map(lambda i: (i, row_number[i]), range(68)))
    
#    def return_row_ind(row_num,ray_object):
#        xous_in_row = [i for i,row in xou_by_row if row == row_num]
#        ind = [ray_object.xou_hit[i] in xous_in_row for i in range(len(ray_object.xou_hit))]
#        return ind
        
#    def return_wave_ind(wavelength,ray_object):
#        ind = [ray_object.wave[i] == wavelength for i in range(len(ray_object.x))]
#        return ind
    
#    # Creating the performance metrics by row:
#    rows = arange(8) + 1
#    ArcusR = zeros((len(rows),len(wavelengths),len(orders)))
#    ArcusEA = zeros((len(rows),len(wavelengths),len(orders)))
    
#    # Scanning the total ray object row-by-row.
#    for i in range(len(rows)):
#        # First, getting all the rays that hit this particular row.
#        print 'Working on rays in Row ' + str(rows[i]) + '....'
#        row_rays = ray_object.yield_object_indices(return_row_ind(rows[i],ray_object))
        
#        # Scanning the row_ray object by wavelength to construct the ArcusEA.
#        for j in range(len(wavelengths)):
#            # Getting all the row_rays matching this wavelength.
#            wave_row_rays = row_rays.yield_object_indices(return_wave_ind(wavelengths[j],row_rays)) 
#            garbage,ArcusEA[i,j] = compute_perf_by_orders(wave_row_rays,illum_area/float(N))

#    print 'Dumping to Pickle File....'
#    f = open(pickle_path + 'RowByRow/' + fileend + '_RowByRow.pk1','wb')
#    cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#    f.close()
    
#    print 'Loading from Pickle File....'
#    f = open(pickle_path + 'RowByRow/' + fileend + '_RowByRow.pk1','rb')
#    [wavelengths,orders,ArcusR,ArcusEA] = cPickle.load(f)
#    f.close()
#    return ArcusR,ArcusEA

###################################################################

def ArcusConfigPerfCalc(opt_chans,fpa,wavelengths,N,\
                        orders= range(0,13,1), home_directory = os.getcwd(), fileend = 'Default',\
                        csv_description = 'Effective Areas and Resolving Powers by Order for Arcus Configuration, 12/02/17',\
                        ea_title_description = '4 Channels (Traced Separately),\nCoated Plates, 6 nm SiC + 10 nm Ir atop Si,\n Full Arcus Focal Plane, No Alignment/Jitter Errors'):
    
    # First, creating the directory structure to output results in the form of plots, .csv files, and
    # pickled files.
    print '#'*40
    print 'Making Results Directories....'
    print '#'*40 + '\n'
    csv_path,plot_path,pickle_path = ArcMCPlot.make_results_directories(home_directory)

    # Performing the Monte Carlo simulation.
    print '#'*40
    print 'Performing the Monte Carlo Calculation....'
    print '#'*40 + '\n' 
    ArcusR, ArcusEA = ArcusMCTrace(opt_chans,fpa,wavelengths,N,orders,fileend,pickle_path)
    
    ## Plotting the effective area of the calculation for the whole configuration.
    #print '#'*40
    #print 'Plotting the effective area for the whole channel...'
    #print '#'*40 + '\n'
    #ArcMCPlot.plot_ea(wavelengths,orders,ArcusR,ArcusEA, plot_fn = plot_path + fileend, title_description = ea_title_description)

    ## Plotting the effective area of the calculation on a channel-by-channel basis.
    #print '#'*40
    #print 'Plotting/outputting the effective areas on a channel-by-channel basis...'
    #print '#'*40 + '\n'
    #ArcMCPlot.do_channel_outputs(wavelengths,orders,ArcusR,ArcusEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description)
    
    ## Plotting the effective area of the calculation on a row-by-row basis.
    #print '#'*40
    #print 'Plotting/outputting the effective areas on a row-by-row basis...'
    #print '#'*40 + '\n'
    #ArcusRowR,ArcusRowEA = do_rowbyrow_recalc(wavelengths,orders,N,fileend,pickle_path)
    #ArcMCPlot.do_rowbyrow_outputs(wavelengths,orders,ArcusRowR,ArcusRowEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description)
    return 
    

