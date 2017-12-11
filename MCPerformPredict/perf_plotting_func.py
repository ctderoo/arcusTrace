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

def plot_res(wavelengths,orders,ChanRes,ChanEA,plot_fn = '171120_arcusTrace_ResPlot_Uncoated.png'):
    #####################################
    # Resolving Power Plot.
    plt.figure(figsize = (12,12))
    plt.plot(wavelengths*10**7,calculate_avg_res(wavelengths,orders,ChanRes,ChanEA))
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Resolution (lambda / Delta lambda)')
    plt.title('Arcus Calculated Average Resolution,\n4 Channels (Traced Separately),\nUncoated (1 nm SiO2, 4 Angstrom Roughness, Si substrate),\n Full Arcus Focal Plane, No Alignment/Jitter Errors')
    
    os.chdir(figure_directory)
    plt.savefig(plot_fn)
    plt.close()
    os.chdir(home_directory)

def plot_ea(wavelengths,orders,ChanRes,ChanEA,plot_fn = '171120_arcusTrace_EffAreaByOrder_Uncoated.png',coating_description = 'Uncoated (1 nm SiO2, 4 Angstrom Roughness, Si substrate)'):
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
    plt.suptitle('Arcus Calculated Effective Area,\n4 Channels (Traced Separately),\n' + coating_description + ',\n Full Arcus Focal Plane, No Alignment/Jitter Errors')
    
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

def plot_channel_ea(wavelengths,orders,ChanRes,ChanEA,plot_fn = '171120_arcusTrace_EffAreaByOrder_Uncoated.png',coating_description = 'Uncoated (1 nm SiO2, 4 Angstrom Roughness, Si substrate)'):
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
    plt.suptitle('Arcus Calculated Effective Area,\n4 Channels (Traced Separately),\n' + coating_description + ',\n Full Arcus Focal Plane, No Alignment/Jitter Errors')

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
                coating_description = 'Uncoated (1 nm SiO2, 4 Angstrom Roughness, Si substrate)',\
                row_string = 'Row Number N'):
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
    plt.suptitle('Arcus Calculated Effective Area,\n4 Channels (Traced Separately),\n' + coating_description + ',\n' + row_string)

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