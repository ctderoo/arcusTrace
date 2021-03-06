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
import arcusTrace.MCPerformPredict.MCPerformanceComputation as ArcMCComp

################################################
# Functions for Exporting to SIXTE
################################################

illum_area = 45*50.

def single_channel_trace_through_gratings(opt_chan,wavelength,order,N,fs_dist = None):
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

    return instrum_chan_rays

def export_energy_to_SIXTE(opt_chans,wavelengths,orders,N,fits_start):
    # Want to trace over each wavelength specified (i), over all orders (j), over all optical channels (k).
    for i in range(len(wavelengths)):
        # This is ultimately the dictionary we'll export to SIXTE.
        instrum_ray_dict = {}
        for j in range(len(orders)):
            for k in range(len(opt_chans)):
                print 'Raytracing OC' + str(k + 1) + ', Order = ' + str(orders[j]) + ', Wavelength = ' + str(wavelengths[i]*10**6) + ' nm...'
                temp_rays = single_channel_trace_through_gratings(opt_chans[k],wavelengths[i],orders[j],N)
                instrum_ray_dict['oc' + str(k) + '_wavestep' + str(i) + '_order' + str(orders[j])] = temp_rays.yield_object_indices(temp_rays.weight > 0)
        # Writing a dictionary every wavelength step.
        print '\nWriting to FITS...\n'
        instrum_ray_object = ArcRays.merge_ray_object_dict(instrum_ray_dict)
        instrum_ray_object.write_to_fits(fits_start + '_E' + "{:5.2f}".format(1240.*10**-6/wavelengths[i]).replace('.','p') + 'eV.fits')



#def single_channel_trace(opt_chan,det_array,wavelength,order,N,fs_dist = None):
#    # Create the source, trace it through the SPO petal and the CAT grating petal.
#    test_rays = ArcRays.make_channel_source(N,order = order,wave = wavelength,fs_dist = fs_dist)
#    spo_petal_rays = ArcSPO.SPOPetalTrace(test_rays,opt_chan.chan_xous)

#    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
#    if len(spo_petal_rays.x) == 0:
#        return spo_petal_rays

#    cat_petal_rays = ArcCAT.CATPetalTrace(spo_petal_rays,opt_chan.chan_facets)

#    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
#    if len(cat_petal_rays.x) == 0:
#        return cat_petal_rays

#    # Do transform to petal coordinates.
#    instrum_chan_rays = copy.deepcopy(cat_petal_rays)
#    opt_chan.rays_from_chan_to_instrum_coords(instrum_chan_rays)
#    instrum_chan_rays.chan_num = ones(len(instrum_chan_rays.x))*opt_chan.chan_num
    
#    if len(instrum_chan_rays.order) != len(instrum_chan_rays.x):
#        pdb.set_trace()

#    # Hit the shared Arcus Focal Plane.
#    det_rays = ArcDet.DetectorArrayTrace(instrum_chan_rays,det_array.fpa_dets)

#    if len(det_rays.order) != len(det_rays.x):
#        pdb.set_trace()

#    # Checking if the ray dictionary is empty. If it is, return the empty dictionary.
#    if len(det_rays.x) == 0:
#        return det_rays

#    # Transform back to petal coordinates for performance evaluation.
#    opt_chan.rays_from_chan_to_instrum_coords(det_rays,inverse = True)
    
#    return det_rays

#def compute_order_res(chan_rays,order,threshold = 0):
#    # If this is zeroth order, there's no spectral resolution.
#    if order == 0:
#        return 0
#    # If not, we want to only consider rays that have made it to the focal plane
#    # i.e. rays with weight > 0.
#    weight_ind = chan_rays.weight > 0
#    # If no rays made it to the focal plane, there's no line -- return spectral
#    # resolution of zero.
#    if sum(weight_ind) == 0:
#        return 0
#    # Else, if this a good order, compute the position and FWHM of the non-zero weight rays,
#    # and return the resolving power estimate.
#    else:
#        try: 
#            x,FWHMx = abs(mean(chan_rays.x[weight_ind])),ArcUtil.compute_FWHM(chan_rays.x[weight_ind])
#        except:
#            pdb.set_trace()
#        return x/FWHMx

####### Doesn't need order!
#def compute_order_EA(chan_rays,order,convert_factor):
#    return sum(chan_rays.weight)*convert_factor

#def ArcusMCTrace(opt_chans,fpa,wavelengths,N,orders,fileend,pickle_path):    
#    # Scanning across optical channel (i), wavelength (j), and order (k).

#    # 05/12/20 -- depreciating the ability to save all rays. Moving to ray weighting (i.e. saving all rays) is creating serious
#    # memory problems.
#    ArcusR,ArcusEA = zeros((len(opt_chans),len(wavelengths),len(orders))),zeros((len(opt_chans),len(wavelengths),len(orders)))

#    #instrum_ray_dict = {}
#    for i in range(len(opt_chans)):
#        for j in range(len(wavelengths)):
#            for k in range(len(orders)):
#                print 'Computing ray bundle -- energy: ' + "{:5.1f}".format(1240./(wavelengths[j]*10**6)) + ' eV, wavelength: ' + "{:1.3f}".format(wavelengths[j]*10**6) + ' nm, order: ' + str(orders[k]) + ', OC' + str(i + 1)
#                chan_rays = single_channel_trace(opt_chans[i],fpa,wavelengths[j],orders[k],N)
   
#                # Checking if the ray dictionary is empty. If it is, if not chan_rays returns True and we scrap that measurement.
#                if len(chan_rays.x) == 0:
#                    ArcusR[i,j,k],ArcusEA[i,j,k] = 0,0
#                else:
#                    ArcusR[i,j,k],ArcusEA[i,j,k] = compute_order_res(chan_rays,orders[k]),compute_order_EA(chan_rays,orders[k],convert_factor = illum_area/float(N))
               
#                ## Moving the rays back to the global instrument coordinates, and adding them to the total dictionary.
#                #try:
#                #    opt_chans[i].rays_from_chan_to_instrum_coords(chan_rays)
#                #except:
#                #    pdb.set_trace()
#                #instrum_ray_dict['oc' + str(i) + '_wavestep' + str(j) + '_order' + str(orders[k])] = chan_rays

#            if j % 20 == 0:
#                #dict_for_merge = copy.deepcopy(instrum_ray_dict)
#                #instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
#                f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
#                print '\n' + 'Dumping to pickle file....' + '\n'
#                cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#                f.close()
#                #instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','Rays.pk1'))
   
#    #dict_for_merge = copy.deepcopy(instrum_ray_dict)
#    #instrum_ray_object = ArcRays.merge_ray_object_dict(dict_for_merge)
#    print '\n' + 'Dumping to pickle file....' + '\n'
#    f = open(pickle_path + '/IndividualChannels/' + fileend + '.pk1','wb')
#    cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#    f.close()
#    #instrum_ray_object.pickle_me(pickle_path + '/Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1'))
#    return ArcusR,ArcusEA

#def do_rowbyrow_recalc(opt_chans,wavelengths,orders,N,fileend,pickle_path):
#    # Loading the rays from the _Rays.pk1 pickle file.
#    print 'Loading rays from pickle file....'
#    path = pickle_path + 'Rays/' + fileend + '.pk1'.replace('.pk1','_Rays.pk1')
#    ray_object = ArcRays.load_ray_object_from_pickle(path)
#    print 'Loaded!'

#    # I'm not sure this is working, but it'll give me a chance to mess around.
#    pdb.set_trace()

#    ###################################
#    # Establishing functions for row mapping.
#    # Row number by XOU.
#    #row_number = repeat(arange(0,6),8)
#    #xou_by_row = tuple(map(lambda x: (opt_chans[0].chan_xous[x].xou_num, opt_chans[0].chan_xous[x].row_num), opt_chans[0].chan_xous.keys()))
#    xou_object = opt_chans[0].chan_xous
#    xou_keys = xou_object.keys()

#    def return_row_ind(row_num,ray_object):
#        xous_in_row = [xou_object[key].xou_num for key in xou_keys if xou_object[key].row_num == row_num]
#        ind = [ray_object.xou_hit[i] in xous_in_row for i in range(len(ray_object.xou_hit))]
#        return ind
        
#    def return_wave_ind(wavelength,ray_object):
#        ind = [ray_object.wave[i] == wavelength for i in range(len(ray_object.x))]
#        return ind

#    def return_order_ind(order,ray_object):
#        ind = [ray_object.order[i] == order for i in range(len(ray_object.x))]
#        return ind
    
#    # Creating the performance metrics by row:
#    rows = arange(6)
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
#            for k in range(len(orders)):
#                order_wave_row_rays = wave_row_rays.yield_object_indices(return_order_ind(orders[k],wave_row_rays))
#                ArcusEA[i,j,k] = compute_order_EA(chan_rays,orders[k],convert_factor = illum_area/float(N))

#    print 'Dumping to Pickle File....'
#    f = open(pickle_path + 'RowByRow/' + fileend + '_RowByRow.pk1','wb')
#    cPickle.dump([wavelengths,orders,ArcusR,ArcusEA],f)
#    f.close()
    
#    print 'Loading from Pickle File....'
#    f = open(pickle_path + 'RowByRow/' + fileend + '_RowByRow.pk1','rb')
#    [wavelengths,orders,ArcusR,ArcusEA] = cPickle.load(f)
#    f.close()
#    return ArcusR,ArcusEA

####################################################################

#def ArcusConfigPerfCalc(opt_chans,fpa,wavelengths,N,\
#                        orders= range(0,13,1), home_directory = os.getcwd(), fileend = 'Default',\
#                        csv_description = 'Effective Areas and Resolving Powers by Order for Arcus Configuration, 12/02/17',\
#                        ea_title_description = '4 Channels (Traced Separately),\nCoated Plates, 6 nm SiC + 10 nm Ir atop Si,\n Full Arcus Focal Plane, No Alignment/Jitter Errors'):
    
#    # First, creating the directory structure to output results in the form of plots, .csv files, and
#    # pickled files.
#    print '#'*40
#    print 'Making Results Directories....'
#    print '#'*40 + '\n'
#    csv_path,plot_path,pickle_path = ArcMCPlot.make_results_directories(home_directory)

#    # Performing the Monte Carlo simulation.
#    print '#'*40
#    print 'Performing the Monte Carlo Calculation....'
#    print '#'*40 + '\n' 
#    ArcusR, ArcusEA = ArcusMCTrace(opt_chans,fpa,wavelengths,N,orders,fileend,pickle_path)
    
#    # Plotting the effective area of the calculation for the whole configuration.
#    print '#'*40
#    print 'Plotting the effective area for the whole channel...'
#    print '#'*40 + '\n'
#    ArcMCPlot.plot_ea(wavelengths,orders,ArcusR,ArcusEA, plot_fn = plot_path + fileend, title_description = ea_title_description)

#    # Plotting the effective area of the calculation on a channel-by-channel basis.
#    print '#'*40
#    print 'Plotting/outputting the effective areas on a channel-by-channel basis...'
#    print '#'*40 + '\n'
#    ArcMCPlot.do_channel_outputs(wavelengths,orders,ArcusR,ArcusEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description)

#    # Need to fix this at a later time -- right now, it relies on the ability to reload rays (which is depreciated).    
#    ## Plotting the effective area of the calculation on a row-by-row basis.
#    #print '#'*40
#    #print 'Plotting/outputting the effective areas on a row-by-row basis...'
#    #print '#'*40 + '\n'
#    #try:
#    #    ArcusRowR,ArcusRowEA = do_rowbyrow_recalc(opt_chans,wavelengths,orders,N,fileend,pickle_path)
#    #    ArcMCPlot.do_rowbyrow_outputs(wavelengths,orders,ArcusRowR,ArcusRowEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description)
#    #except:
#    #    pdb.set_trace()

#    return 
    

