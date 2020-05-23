from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pdb
import pickle
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
import arcusTrace.arcusRays as ArcRays

home_directory = os.getcwd()
figure_directory = home_directory + '/ResultFigures'

def wavelength_ind_hunt(wavelengths,low,high):
    return logical_and(wavelengths > low*1e-6,wavelengths < high*1e-6)

def avg_EA_over_range(summed_EA,wavelengths,low,high):
    return mean(summed_EA[wavelength_ind_hunt(wavelengths,low,high)])

def calculate_avg_res(wavelengths,orders,ChanRes,ChanEA):
    EA_summed = sum(ChanEA,axis = 0)
    EA_norm = sum(EA_summed[:,1:],axis = 1)
    avg_res = zeros(shape(ChanRes)[1])
    for i in range(len(wavelengths)):
        if EA_norm[i] != 0:
            avg_res[i] = sum(ChanRes[:,i,:]*ChanEA[:,i,:])/EA_norm[i]
        else:
            pass
    return avg_res

def compute_perf_by_orders(chan_rays,convert_factor,all_orders = range(0,13,1)):
    res = asarray([compute_order_res(chan_rays,order) for order in all_orders])
    ea = asarray([compute_order_EA(chan_rays,order,convert_factor) for order in all_orders])
    return res,ea

def plot_merit(wavelengths,orders,ChanRes,ChanEA,plot_fn = 'default_merit.png',title_description = 'description_error'):
    #####################################
    # Resolving Power Plot.
    plt.figure(figsize = (12,12))
    merit = ChanRes*ChanEA
    merit_by_channel = sum(merit,axis = 0) # Summing over channels
    merit_by_order = sum(merit_by_channel,axis = 1) # Summing over orders

    plt.plot(wavelengths*10**7,sqrt(merit_by_order))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Sqrt(R * EA)')
    plt.title('Arcus Calculated Weak Line Detection Merit Function\n' + title_description)
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)
    
def plot_res(wavelengths,orders,ChanRes,ChanEA,plot_fn = 'default_res.png',title_description = 'description_error'):
    #####################################
    # Resolving Power Plot.
    plt.figure(figsize = (12,12))
    plt.plot(wavelengths*10**7,calculate_avg_res(wavelengths,orders,ChanRes,ChanEA))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Resolution (lambda / Delta lambda)')
    plt.title('Arcus Calculated Average Resolution,\n' + title_description)
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)

def plot_ea(wavelengths,orders,ChanRes,ChanEA,plot_fn = 'default.png',title_description = 'description_error'):
    ea_all_channels = sum(ChanEA,axis = 0)
    spectral_ea = sum(ea_all_channels[:,1:],axis = 1)
    
    #####################################
    # Effective Area By Order Plot.
    plt.figure(figsize = (12,12))
    ax = plt.gca()
    
    # Plotting the spectral effective area.    
    plt.plot(wavelengths*10**7,spectral_ea,'k-',label = 'Total')

    # Setting up the plotting "by order".
    order_start = 0
    eff_area_order_names = ['0th','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th']
    eff_area_colors = []
    
    num_colors = 12
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/num_colors) for i in range(num_colors)])
    
    # Performing the plotting by order.
    for i in range(orders[order_start],orders[-1] + 1,1):
        if orders[i] == 0:
            plt.plot(wavelengths*10**7,ea_all_channels[:,i],linestyle = 'dotted',color = 'k',label = eff_area_order_names[i] + ' Ord.') 
        else:
            plt.plot(wavelengths*10**7,ea_all_channels[:,i],linestyle = 'dashed',label = eff_area_order_names[i] + ' Ord.') 
    plt.legend(ncol = 3,loc = 'upper right')
    
    # Labeling the plot, and setting the ticks consistently. 
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Effective Area (sq. cm)')
    plt.suptitle('Arcus Calculated Effective Area,\n' + title_description)
    
    plt.ylim(0,600)
    x1,x2 = plt.xlim()
    ax.set_xticks(arange(plt.xticks()[0][0],plt.xticks()[0][-1], 5), minor=True)
    plt.xlim(x1,x2)
    ax.set_yticks(arange(0, 601, 100))
    ax.set_yticks(arange(0, 601, 50), minor=True)   
    
    plt.grid(which = 'major',alpha = 0.4,linestyle = 'dashed')
    plt.grid(which = 'minor',alpha = 0.2,linestyle = 'dashed')
    
    plt.text(0.03,0.97,'Average Effective Area:',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.94,'16 - 21.6 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,1.6,2.16)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.91,'21.6 - 28 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,2.16,2.8)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.88,'33.7 - 40 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,3.37,4.00)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)

def plot_channel_ea(wavelengths,orders,ChanRes,ChanEA,plot_fn = '171120_arcusTrace_EffAreaByOrder_Uncoated.png',title_description = 'Default'):
    ea_channel = ChanEA
    spectral_ea = sum(ea_channel[:,1:],axis = 1)
    
    #####################################
    # Effective Area By Order Plot.
    plt.figure(figsize = (12,12))
    ax = plt.gca()
    
    # Plotting the spectral effective area.    
    plt.plot(wavelengths*10**7,spectral_ea,'k-',label = 'Total')

    # Setting up the plotting "by order".
    order_start = 0
    eff_area_order_names = ['0th','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th']
    eff_area_colors = []
    
    num_colors = 12
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/num_colors) for i in range(num_colors)])
    
    # Performing the plotting by order.
    for i in range(orders[order_start],orders[-1] + 1,1):
        if orders[i] == 0:
            plt.plot(wavelengths*10**7,ea_channel[:,i],linestyle = 'dotted',color = 'k',label = eff_area_order_names[i] + ' Ord.') 
        else:
            plt.plot(wavelengths*10**7,ea_channel[:,i],linestyle = 'dashed',label = eff_area_order_names[i] + ' Ord.') 
    plt.legend(ncol = 3,loc = 'upper right')
    
    # Labeling the plot, and setting the ticks consistently. 
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Effective Area (sq. cm)')
    plt.suptitle('Arcus Calculated Effective Area,\n' + title_description)

    plt.ylim(0,150)
    x1,x2 = plt.xlim()
    ax.set_xticks(arange(plt.xticks()[0][0],plt.xticks()[0][-1], 5), minor=True)
    plt.xlim(x1,x2)
    ax.set_yticks(arange(0, 151, 25))
    ax.set_yticks(arange(0, 151, 5), minor=True)   
    
    plt.grid(which = 'major',alpha = 0.4,linestyle = 'dashed')
    plt.grid(which = 'minor',alpha = 0.2,linestyle = 'dashed')
    
    plt.text(0.03,0.97,'Average Effective Area:',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.94,'16 - 21.6 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,1.6,2.16)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.91,'21.6 - 28 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,2.16,2.8)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.88,'33.7 - 40 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,3.37,4.00)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)

def plot_row_ea(wavelengths,orders,RowEA,plot_fn = '171202_arcusTrace_EffAreaByOrder_Uncoated_RowN.png',\
                title_description = 'Default',row_string = 'Row Number N'):
    spectral_ea = sum(RowEA[:,1:],axis = 1)
    
    #####################################
    # Effective Area By Order Plot.
    plt.figure(figsize = (12,12))
    ax = plt.gca()
    
    # Plotting the spectral effective area.    
    plt.plot(wavelengths*10**7,spectral_ea,'k-',label = 'Total')

    # Setting up the plotting "by order".
    order_start = 0
    eff_area_order_names = ['0th','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th']
    eff_area_colors = []
    
    num_colors = 12
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/num_colors) for i in range(num_colors)])
    
    # Performing the plotting by order.
    for i in range(orders[order_start],orders[-1] + 1,1):
        if orders[i] == 0:
            plt.plot(wavelengths*10**7,RowEA[:,i],linestyle = 'dotted',color = 'k',label = eff_area_order_names[i] + ' Ord.') 
        else:
            plt.plot(wavelengths*10**7,RowEA[:,i],linestyle = 'dashed',label = eff_area_order_names[i] + ' Ord.') 
    plt.legend(ncol = 3,loc = 'upper right')
    
    # Labeling the plot, and setting the ticks consistently. 
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Effective Area (sq. cm)')
    plt.suptitle('Arcus Calculated Effective Area,\n' + title_description + ',\n' + row_string)

    plt.ylim(0,100)
    x1,x2 = plt.xlim()
    ax.set_xticks(arange(plt.xticks()[0][0],plt.xticks()[0][-1], 5), minor=True)
    plt.xlim(x1,x2)
    ax.set_yticks(arange(0, 101, 10))
    ax.set_yticks(arange(0, 101, 5), minor=True)   
    
    plt.grid(which = 'major',alpha = 0.4,linestyle = 'dashed')
    plt.grid(which = 'minor',alpha = 0.2,linestyle = 'dashed')
    
    plt.text(0.03,0.97,'Average Effective Area:',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.94,'16 - 21.6 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,1.6,2.16)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.91,'21.6 - 28 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,2.16,2.8)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    plt.text(0.03,0.88,'33.7 - 40 Ang. = ' + "{:4.1f}".format(avg_EA_over_range(spectral_ea,wavelengths,3.37,4.00)) + ' sq. cm.',ha = 'left',transform = ax.transAxes)
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)

def write_to_csv_file(wavelengths,orders,ChanRes,ChanEA,csv_file_name = 'default.csv',result_file = 'default.pk1',\
                      description_line = 'Effective Areas and Resolving Powers by Order for Arcus Configuration, 11/22/17'):
    order_names = ['0th','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th']
    column_names = ['wave'] + ['EA_' + order_names[i] + 'Order' for i in range(len(order_names))] + ['Res_' + order_names[i] + 'Order' for i in range(len(order_names))]
    
    column_units = ['nm'] + ['-' for i in range(len(order_names))] + ['-' for i in range(len(order_names))]
    column_datatypes = ['float64'] + ['float64' for i in range(len(order_names))] + ['float64' for i in range(len(order_names))]
    
    with open(csv_file_name,'wb') as csvfile:
        csvfile.write('# %ECSV 0.9\n')
        csvfile.write('# ---\n')
        csvfile.write('# datatype:\n')
        
        for m in range(len(column_names)):
            string_to_write = '# - {name: ' + column_names[m]
            if column_units[m] is not '-':
                string_to_write += ',' + 'unit: ' + column_units[m]
            if column_datatypes[m] is not '-':
                string_to_write += ',' + 'datatype: ' + column_datatypes[m]  
            string_to_write += '}\n'
            csvfile.write(string_to_write)
        
        csvfile.write('# Author: Casey DeRoo\n')
        csvfile.write('# ORIGFILE: ' + result_file + '\n')
        csvfile.write('# Description: ' + description_line + '\n')
        
        writer = csv.writer(csvfile,delimiter = ',')
        writer.writerow(column_names)
        for n in range(len(wavelengths)):
            line_to_write = [wavelengths[n]*10**6] + list(ChanEA[n,:]) + list(ChanRes[n,:])
            writer.writerow(line_to_write)
    
    csvfile.close()
    return

################################################
# Functions for Housekeeping
################################################

def make_results_directories(home_directory):
    if not os.path.exists(home_directory + '/ResultCSVFiles'):
        os.makedirs(home_directory + '/ResultCSVFiles')
        os.makedirs(home_directory + '/ResultCSVFiles/IndividualChannels')
        os.makedirs(home_directory + '/ResultCSVFiles/RowByRow')
    
    if not os.path.exists(home_directory + '/ResultFigures'):
        os.makedirs(home_directory + '/ResultFigures')
        os.makedirs(home_directory + '/ResultFigures/IndividualChannels')
        os.makedirs(home_directory + '/ResultFigures/RowByRow')
        
    if not os.path.exists(home_directory + '/ResultPickles'):
        os.makedirs(home_directory + '/ResultPickles')
        os.makedirs(home_directory + '/ResultPickles/IndividualChannels')
        os.makedirs(home_directory + '/ResultPickles/RowByRow')
        
    csv_path = home_directory + '/ResultCSVFiles/'
    plot_path = home_directory + '/ResultFigures/'
    pickle_path = home_directory + '/ResultPickles/'
    
    return csv_path,plot_path,pickle_path

def do_channel_outputs(wavelengths,orders,ArcusR,ArcusEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description):
    for n in range(len(ArcusR)):
        write_to_csv_file(wavelengths,orders,ArcusR[n],ArcusEA[n],csv_file_name = csv_path + 'IndividualChannels/' + fileend + '_OC' + str(n + 1) + '.csv',\
                            result_file = fileend + '.pk1', description_line = csv_description)
        plot_channel_ea(wavelengths,orders,ArcusR[n],ArcusEA[n],\
                            plot_fn = plot_path + 'IndividualChannels/' + fileend + '_OC' + str(n + 1) + '.png',\
                            title_description = ea_title_description)
    return

def do_rowbyrow_outputs(wavelengths,orders,rowR,rowEA,fileend,csv_path,pickle_path,plot_path,csv_description,ea_title_description):
    print 'Making row-by-row .csv files and plots....'

    ArcusRowEA = sum(rowEA,axis = 1)    # Indices for rowEA are row, optical channel, wavelength order. So here we sum over optical channel
    
    ## Is this right? Not a clue.
    #pdb.set_trace()
    # Averaging resolution over all four channels, weighted by the counts in them.
    weight_R = sum(rowR*rowEA,axis = 1)/ArcusRowEA
    ArcusRowR = divide(weight_R,ArcusRowEA,out=zeros_like(weight_R),where=ArcusRowEA!=0)
    for i in range(len(ArcusRowR)):
        write_to_csv_file(wavelengths,orders,ArcusRowR[i],ArcusRowEA[i],csv_file_name = csv_path + 'RowByRow/' + fileend + '_Row' + str(i + 1) + '.csv', result_file = pickle_path + 'RowByRow/' + fileend + '_RowByRow.pk1',description_line = csv_description)
        plot_row_ea(wavelengths,orders,ArcusRowEA[i],plot_fn = plot_path + 'RowByRow/' + fileend + '_Row' + str(i + 1) + '.png', title_description = ea_title_description, row_string = 'Row Number ' + str(i + 1))
    return

