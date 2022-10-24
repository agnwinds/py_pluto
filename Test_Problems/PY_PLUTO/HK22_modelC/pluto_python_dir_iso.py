#!/usr/bin/env python

import subprocess,sys
import glob
from astropy import constants as c
from astropy import units as u
from scipy.integrate import quad
import pyPLUTO as pp
import numpy as np
import pluto_python_sub as pps
import re



#This imports the definitions.h file - it is needed because pluto expects geometrical parameters to be in cude units, and the translation between code and physical units is set by the UNIT_ commands.

try:
    UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY=pps.get_units()
except:
    print("Unable to open definitions.h file - big problems")
    exit()


disk_rad_eff=0.06 #disk reprocessing efficiency
eddington_ratio=0.5


data={} #Set up the dictionary that will contain all of the information for the run


#This sets up the number of cores to use for the three different codes. Python and CAK are automatically mpi - but pluto needs to be specifcially compiled that way.

data["nproc_py"]=1
data["nproc_cak"]=1
data["nproc_pluto"]=1



#First, set up the grid - these are in physical units, the imported definitions.h file will convert to code units to make the pluto.ini
#I've only set up an r-theta grid so far..

#radial bins
data["R_MIN"]=8.7e8
data["R_MAX"]=87.e8
data["N_R"]=128

#Rescale the grid
data["R_MIN"]=data["R_MIN"]/UNIT_LENGTH
data["R_MAX"]=data["R_MAX"]/UNIT_LENGTH

#theta bins
data["T_MIN"]=np.radians(0.0)
data["T_MAX"]=np.radians(90.0)
data["N_T"]=96


#some things not used in these runs - to do with heating and cooling.


data["T_x"]=160000. #This is not used in these runs - when heating and cooling are implemented via blondins formulae it is needed
data["f_uv"]=0.9
data["f_x"]=0.1 #This is the proportion of disk flux we assume comes from the middle as X-rays
data["L_star"]=9.05e34 #The luminosity of the central source




#Set the parameters of the run here

data["rad_force"]=1 #this turns on bits of the code that deals with radiation - not quite sure what would happen if it was turned off!
data["RHO_0"]=1e-9    #The midplane density
data["RHO_ALPHA"]=0.0  #The dropoff of the midplane density with radius - not used in these runs
data["R_0"]=8.31e17 #The scaling radius if we are dropping off with radius - so rho_midplane = rho_0 at r_0 - with rho_alpha=0 this does nothing.
data["MU"]=0.6 #The mean molecular mass - used to compute things like sound speed
data["T_ISO"]=40000. #The isothermal temperature 

#Things mainly used for the python part of the run

data["system_type"]='star' #/The overall geometry type


data["CENT_RADIUS"]=8.7e8 #radius of the central object
data["CENT_MASS"]=0.6*c.M_sun.cgs.value #mass of the central object

#We are using a boundary layer - because we can set the luminosity and temperature seperately

data["cent_spectype"]="none" #Turns off the standard python central source
data["boundary_layer"]="yes" #turns on the boundary layer
data["L_BL"]=9.05e+34 #boundary layer luminosity
data["T_BL"]=40000. #boundsry layer temperature


#More technical Stuff for the python runs

data["python_ver"]="86b"  #the version of python to use - this was written to work with py86b - if it doesnt run - modifications may need to be made to the "python_input_file" subroutine in the pluto_python_sub.py file.

data["NPHOT"]=1e7 #number of photons

data["disk_radiation"]="yes" #turn on disk radiation
data["PY_DISK_MDOT"]=3.14e-8 #and set the rate in solar masses per year
data["DISK_MDOT"]=data["PY_DISK_MDOT"]*c.M_sun.cgs.value/60/60/24/365.25 #we also need the disk accretion rate in g/s for pluto to compute the disk temperature
data["DISK_TRUNC_RAD"]=8.7e9 #The maximum radius occupied by the disk - the odd name is a throwback to older interations



data["wind_radiation"]="yes" #turn on the wind

data["line_trans"]="simple" #line transfer mode - allows the potential for macro atom usage.



#Set up the initial things for the run

t0=1.0 #The run time for the initial pluto run - the first run is to produce a starting geometry
dt=2.0   #The time between calls to pluto (in seconds)

if t0==0.0:
    print ("We need to run for at least one second, dummy")
    t0=1.0
init_py_cycles=5 #This is the initial number of cycles that python will do - worth doing a few more here than we will in future to get a bit of convergence in ionization state 
py_cycles=2 #once we have started - we will do this number of cycles in python - should really be a minimum of 2.




logfile=open("run_logfile","w")

#these lines allow for restarts - if there is a numerical argument after the command line ./pluto_python_dir_iso.py <number> it will try to restart from the files contained in the cycle_<number> directory


if len(sys.argv)<2:
    istart=0
    time=t0
else:
    istart=int(sys.argv[1])
    if istart>0:
        root="%08d"%(istart)
        directory="cycle"+root
        print ("We will be trying to restart using files in "+directory)
        try:
            subprocess.check_call("cp "+directory+"/* .",shell=True)
        except:
            print ("Cannot restart")
            exit(0)
        print ("Last run finished at ",pp.pload.pload(istart).SimTime)
        time=pp.pload.pload(istart).SimTime+dt
    else:
        istart=0
        time=t0


#In the absence of force multiplier files - we use a k-alpha formulation for the initial pluto run - the values to use are set here

data["k"]=0.59
data["alpha"]=-0.6 #NB - alpha is normally negative.


#We need fluxes in each cell to drive the flow. This code expects some files with optcially thin fluxes generated from python to exist to be copied into the files pluto is expecting to ingest. These next three lines copy over the stored files.

subprocess.check_call("cp directional_flux_x_opt_thin.dat directional_flux_x.dat",shell=True)   
subprocess.check_call("cp directional_flux_y_opt_thin.dat directional_flux_y.dat",shell=True)   
subprocess.check_call("cp directional_flux_z_opt_thin.dat directional_flux_z.dat",shell=True)   




for i in range(istart,1000): #Do a loop from istart to some big number. You can set the big number to anything - but if something goes wrong, it should be limited to avoid spamming loads of files that could jam things up..
    if i>0: #If we are restarting - then we signal to pluto that it should be importaing force multiplier - the next two numbers act as that signal
        data["k"]=999
        data["alpha"]=999
        
        
    root="%08d"%(i) #We want files generated to be named in a way that is easy to see in an ls
    logfile.write("Making a pluto input file for cycle "+str(i)+"\n")
    print("Making a pluto input file for cycle "+str(i)+"\n")
    
    pps.pluto_input_file(time,data) #This makes a pluto.ini file then the next few lines run pluto - either restarting, or from scratch and using mpi if necessary. The lines construct an argument to be sent to subprocess which actually runs the command
    
    if i==0: 
        if data["nproc_pluto"]==1:
            cmdline="./pluto > pluto_log"
        else:
            cmdline="mpirun -n "+str(data["nproc_pluto"])+" ./pluto > pluto_log" #we run with MPI if we have asked for multiple cores for pluto
        
    else:
        if data["nproc_pluto"]==1:
            cmdline="./pluto -restart "+str(i)+" > pluto_log"
        else:
            cmdline="mpirun -n "+str(data["nproc_pluto"])+" ./pluto -restart "+str(i)+" > pluto_log"

    logfile.write("Running pluto run"+"\n")
    print("Running pluto run"+"\n")
    
    logfile.write("command line: "+cmdline+"\n")
    print("command line: "+cmdline+"\n")
    
    subprocess.check_call(cmdline,shell=True) #This actually runs the code
    logfile.write("Finished pluto run"+"\n")
    print("Finished pluto run"+"\n")
    

   

    ifile=i+1   #The current pluto save file will by one further on....
    time=time+dt #The next pluto run will need to run a little longer
    root="%08d"%(ifile) #This is the root we will use for all file associated with this pluto save file
    dbl="data."+"%04d"%(ifile)+".dbl"  #The name of the dbl file 
    directory="cycle"+root #The name of the directory we will save all the files into

    logfile.write("Turning dbl file "+dbl+" into a python model file"+"\n")
    
    print("Turning dbl file "+dbl+" into a python model file"+"\n")
    
    
#Make a python model file from pluto output    
    py_model_file=pps.pluto2py_rtheta(ifile)    
    logfile.write("Made a python model file called "+py_model_file+"\n")
    print("Made a python model file called "+py_model_file+"\n")

    
#Make a python parameter file to carry out a simulation on the    
    logfile.write("Making a python input file"+"\n")
    print("Making a python input file"+"\n")    
    pps.python_input_file(root,data,cycles=init_py_cycles+i*py_cycles)
    logfile.write("Successfully made a python input file"+"\n")
    print("Successfully made a python input file"+"\n")


#Copy the python file to a generic name so windsave files persist  
    cmdline="cp "+root+".pf input.pf"
    logfile.write("command line: "+cmdline+"\n")
    print("command line: "+cmdline+"\n")    
    subprocess.check_call(cmdline,shell=True)  

#The next chunk deals with running python, 

    if i>0: #If we are restarting
        logfile.write("restarting python"+"\n")
        
        #Firstly we paint the new densities over old windsave - this allows the ionization state to be preserved but uses the new densities and velocities
        logfile.write("running modify wind on the old windsave"+"\n")        
        cmdline="modify_wind"+data["python_ver"]+" -model_file "+py_model_file+" input > mod_wind_log" 
        logfile.write("command line: "+cmdline+"\n")
        subprocess.check_call(cmdline,shell=True) 
        cmdline="cp new.wind_save input.wind_save"
        logfile.write("command line: "+cmdline+"\n")          
        subprocess.check_call(cmdline,shell=True) 
        #now we run python - restarting, we run in classic mode - probably not a problem with CVs, but we are unsure about how to deal with relativity on fluxes etc, so for consistency we are doing classic
        cmdline="mpirun -n "+str(data["nproc_py"])+" py"+data["python_ver"]+" -f -r -classic input.pf > python_log"        
    else:   
        cmdline="mpirun -n "+str(data["nproc_py"])+" py"+data["python_ver"]+" -f -classic input.pf > python_log"
        
    logfile.write("Running python"+"\n")
    logfile.write("command line: "+cmdline+"\n")
    print("Running python"+"\n")
    print("command line: "+cmdline+"\n")          
    subprocess.check_call(cmdline,shell=True)
    logfile.write("Finished python"+"\n")
    print("Finished python"+"\n")
    
#Now we set about processing the python outputs into what we need to inform pluto about fluxes etc. This is all handled by rad_hydro_files - a routine complied as part of the python districution.

    cmdline="rad_hydro_files"+data["python_ver"]+" input > rad_hydro_files_output" 
    logfile.write("command line: "+cmdline+"\n") 
    print("command line: "+cmdline+"\n") 
    subprocess.check_call(cmdline,shell=True)
    
    
#Copy the resulting files into versions that wont get overwritten by future cycles.    

    cmdline="cp py_heatcool.dat "+root+"_py_heatcool.dat"  
    subprocess.check_call(cmdline,shell=True)   #And finally we take a copy of the python heatcool file for later 
    cmdline="cp py_driving.dat "+root+"_py_driving.dat"  
    subprocess.check_call(cmdline,shell=True)   #And finally we take a copy of the python heatcool file for later 
    cmdline="cp py_pcon_data.dat "+root+"_py_pcon_data.dat"  
    subprocess.check_call(cmdline,shell=True)   #And finally we take a copy of the python heatcool file for later 


#This subroutine takes the accelerations from electron scattering and bound free and makes them into a file that pluto can ingest.  
 
    pps.accel_calc(ifile) 

#These lines run the external cak_v3 code which computes force multiplier as a function of t for each cell

    cmdline="mpirun -n "+str(data["nproc_cak"])+" ./cak_v3 > cak_output"
    print ("Running CAK")
    print (cmdline)
    subprocess.check_call(cmdline,shell=True)   #And finally we take a copy of the python heatcool file for later 
    print ("Finished CAK")
    


#This cleans up a bit - by removing the directory containing restart files from a couple of cycles ago. If you want to keep all info from all cycles, comment this out - but it takes up a lot of disk space pretty fast..    
     
    subprocess.check_call("rm -rf cycle"+"%08d"%(ifile-2),shell=True)        
    
    
#copy all the generated files into a storage directory - for analysis and restarts.    
    
    try:
        subprocess.check_call("mkdir "+directory,shell=True)
    except:
        subprocess.check_call("rm -rf "+directory+"_old",shell=True)        
        subprocess.check_call("mv "+directory+" "+directory+"_old",shell=True)
        subprocess.check_call("mkdir "+directory,shell=True)
    
    subprocess.check_call("cp dbl.out "+directory,shell=True) #Copy the model file to the storage directory
    subprocess.check_call("cp pluto.ini "+directory,shell=True) #Copy the model file to the storage directory
    subprocess.check_call("cp restart.out "+directory,shell=True) #Copy the model file to the storage directory
    subprocess.check_call("cp "+dbl+" "+directory,shell=True) #Copy the model file to the storage directory
    subprocess.check_call("cp py_*.dat "+directory,shell=True) #Copy the rad_hydro output files to storage directory
    subprocess.check_call("cp M_UV_data.dat "+directory,shell=True) #Copy the CAK output files to storage directory
    
    subprocess.check_call("cp input.wind_save "+directory,shell=True) #Copy the CAK output files to storage directory
    
    subprocess.check_call("mv "+py_model_file+" "+directory,shell=True) #Copy the model file to the storage directory
    subprocess.check_call("mv "+root+".pf "+directory,shell=True)  #And also copy the python file to storage directory
    subprocess.check_call("mv "+root+"_py_heatcool.dat "+directory,shell=True)  
    subprocess.check_call("mv "+root+"_py_driving.dat "+directory,shell=True)  
    subprocess.check_call("mv "+root+"_py_pcon_data.dat "+directory,shell=True)  
    
    #   subprocess.check_call("cp prefactors.dat "+directory,shell=True) #Copy the CAK output files to storage directory
    logfile.write("Finished tidying up"+"\n")
    
    
print ("Fin")
