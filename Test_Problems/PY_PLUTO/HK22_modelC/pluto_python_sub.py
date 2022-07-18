#!/usr/bin/env python

import subprocess
import glob
from astropy.io import ascii
from astropy.table import Table
from astropy import constants as c
from astropy import units as u
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate


import pyPLUTO as pp
import numpy as np


#This routine get code units from the definitions.h file 

def get_units(fname='definitions.h'):
    inp=open('definitions.h','r')
    for line in inp.readlines():
        data=line.split()
        if len(data)>1:
            if data[1]=='UNIT_DENSITY':
                UNIT_DENSITY=float(data[2])
            elif data[1]=='UNIT_LENGTH':
                UNIT_LENGTH=float(data[2])
            elif data[1]=='UNIT_VELOCITY':
                UNIT_VELOCITY=float(data[2])
    inp.close()
    return(UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY)
    
    

#This subroutine makes a pluto input file    

def pluto_input_file(tlim,data):
    output=open('pluto.ini','w')
    output.write("[Grid]\n")
    output.write("\n")
    output.write("X1-grid 1 "+str(data["R_MIN"])+" "+str(data["N_R"])+" r "+str(data["R_MAX"])+" 1.07\n")
    output.write("X2-grid 1 "+str(data["T_MIN"])+" "+str(data["N_T"])+" r "+str(data["T_MAX"])+" 0.95\n")
    output.write("X3-grid 1    0.0    1      u    1.0\n")
    output.write("\n")
    output.write("[Chombo Refinement]\n")
    output.write("\n")
    output.write("Levels           4\n")
    output.write("Ref_ratio        2 2 2 2 2\n") 
    output.write("Regrid_interval  2 2 2 2 \n")
    output.write("Refine_thresh    0.3\n")
    output.write("Tag_buffer_size  3\n")
    output.write("Block_factor     8\n")
    output.write("Max_grid_size    64\n")
    output.write("Fill_ratio       0.75\n")
    output.write("\n")
    output.write("[Time]\n")
    output.write("\n")
    output.write("CFL              0.4\n")
    output.write("CFL_max_var      1.1\n")
    output.write("tstop            "+str(tlim)+"\n")
    output.write("first_dt         1e-4\n")
    output.write("\n")
    output.write("[Solver]\n")
    output.write("\n")
    output.write("Solver         hll\n")
    output.write("\n")
    output.write("[Boundary]\n")
    output.write("\n")
    output.write("X1-beg        reflective\n")
    output.write("X1-end        outflow\n")
    output.write("X2-beg        axisymmetric\n")
    output.write("X2-end        reflective\n")
    output.write("X3-beg        outflow\n")
    output.write("X3-end        outflow\n")
    output.write("\n")
    output.write("[Static Grid Output]\n")
    output.write("\n")
    if data["rad_force"]==1:
        output.write("uservar    21 XI T ch cc lc bc xh ch_pre cc_pre lc_pre bc_pre xh_pre ne nh gr gt gp dv_ds t M_max1 M_max2\n")
    else:
        output.write("uservar    14    XI T ch cc lc bc xh ch_pre cc_pre lc_pre bc_pre xh_pre ne nh\n")
    output.write("dbl        1000000000000   -1   single_file\n")
    output.write("flt       -1.0  -1   single_file\n")
    output.write("vtk       -1.0  -1   single_file\n")
    output.write("dbl.h5    -1.0  -1\n")
    output.write("flt.h5    -1.0  -1\n")
    output.write("tab       -1.0  -1   \n")
    output.write("ppm       -1.0  -1   \n")
    output.write("png       -1.0  -1\n")
    output.write("log        1000\n")
    output.write("analysis  -1.0  -1\n")
    output.write("\n")
    output.write("[Chombo HDF5 output]\n")
    output.write("\n")
    output.write("Checkpoint_interval  -1.0  0\n")
    output.write("Plot_interval         1.0  0 \n")
    output.write("\n")
    output.write("[Parameters]\n")
    output.write("\n")    
    output.write("MU                          %.2f"%(data["MU"])+"\n")
    output.write("RHO_0                       %4.2e"%(data["RHO_0"])+"\n")
    output.write("R_0                         %4.2e"%(data["R_0"])+"\n")
    output.write("RHO_ALPHA                   %.2f"%(data["RHO_ALPHA"])+"\n")
    output.write("CENT_MASS                   %4.2e"%(data["CENT_MASS"])+"\n")
    output.write("DISK_MDOT                   %4.2e"%(data["DISK_MDOT"])+"\n")
    output.write("T_ISO                       %4.2e"%(data["T_ISO"])+"\n")
    output.write("L_star                      %4.2e"%(data["L_star"])+"\n")
    output.write("f_x                         %.2f"%(data["f_x"])+"\n") 
    output.write("f_uv                        %.2f"%(data["f_uv"])+"\n")
    output.write("T_x                         %4.2e"%(data["T_x"])+"\n")    
    output.write("KRAD                        %4.2e"%(data["k"])+"\n")
    output.write("ALPHARAD                    %4.2e"%(data["alpha"])+"\n")
    output.close()
    return
    
    
    
#convert a pluto dbl file into a file that can be read in by python.    
    
def pluto2py_rtheta(ifile):
    D=pp.pload(ifile)
    UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY=get_units('definitions.h')

    #Extract the bits of the pluto file we will be needing - and scale accordinglt
    pluto_r_inner=D.x1r*UNIT_LENGTH #Python expects a model file to have the inner radius of each shell these are the pluto coordinates in the inner edge
    pluto_theta_inner=D.x2r #Theta coordinates of inner edges of cell

    pluto_r_c=D.x1*UNIT_LENGTH #Central point of each pluto cell - where the velocity is defined
    pluto_theta_c=D.x2  #Central theta of each pluto cell - where the velocity is defined

    pluto_vr_c=D.vx1*UNIT_VELOCITY #The radial velocity in each pluto cell - defined at cell centre
    pluto_vt_c=D.vx2*UNIT_VELOCITY #The theta velocity in each pluto cell - defined at cell centre
    pluto_vy_c=D.vx3*UNIT_VELOCITY #The y component is just equal to the phi component....

    pluto_density=D.rho*UNIT_DENSITY
    pluto_temperature=D.T


    #Set up arrays to take the velocities in x,z components, how python expects it
    pluto_vx_c=np.zeros(np.shape(pluto_vr_c))
    pluto_vz_c=np.zeros(np.shape(pluto_vr_c))

    #Set up python r and theta arrays - they are extended a bit to make ghost cells
    python_r=[]
    python_theta=[]

    for i in range(len(pluto_r_inner)):
        python_r.append(pluto_r_inner[i])
    python_r.append(pluto_r_inner[-1]+(pluto_r_inner[-1]-pluto_r_inner[-2]))
    
    for i in range(len(pluto_theta_inner)):
        python_theta.append(pluto_theta_inner[i])
    python_theta.append(pluto_theta_inner[-1]+(pluto_theta_inner[-1]-pluto_theta_inner[-2]))


    #Work out the x and z velocities on the pluto grid from the r theta compnents.
    for i in range(len(pluto_r_c)):
        for j in range(len(pluto_theta_c)):
                pluto_vx_c[i][j]=pluto_vr_c[i][j]*np.sin(pluto_theta_c[j])+pluto_vt_c[i][j]*np.cos(pluto_theta_c[j])
                pluto_vz_c[i][j]=pluto_vr_c[i][j]*np.cos(pluto_theta_c[j])-pluto_vt_c[i][j]*np.sin(pluto_theta_c[j])

    
    #Now we interpolate the python connrdinates on the pluto grid
    
    #First, we interpolate in the r direction
    vx_temp=np.zeros([len(python_r),len(pluto_theta_c)])
    vy_temp=np.zeros([len(python_r),len(pluto_theta_c)])
    vz_temp=np.zeros([len(python_r),len(pluto_theta_c)])

    for i in range(len(pluto_theta_c)):
        vx=interpolate.interp1d(pluto_r_c,pluto_vx_c[:,i],fill_value='extrapolate')
        vy=interpolate.interp1d(pluto_r_c,pluto_vy_c[:,i],fill_value='extrapolate')
        vz=interpolate.interp1d(pluto_r_c,pluto_vz_c[:,i],fill_value='extrapolate')
        vx_temp[:,i]=vx(python_r)
        vy_temp[:,i]=vy(python_r)
        vz_temp[:,i]=vz(python_r)
    

    #And now we interpolate in the theta direction
    python_vx=np.zeros([len(python_r),len(python_theta)])
    python_vy=np.zeros([len(python_r),len(python_theta)])
    python_vz=np.zeros([len(python_r),len(python_theta)])
 
    for i in range(len(python_r)):
        vx=interpolate.interp1d(pluto_theta_c,vx_temp[i],fill_value='extrapolate')
        vy=interpolate.interp1d(pluto_theta_c,vy_temp[i],fill_value='extrapolate')
        vz=interpolate.interp1d(pluto_theta_c,vz_temp[i],fill_value='extrapolate')
        python_vx[i]=vx(python_theta)
        python_vy[i]=vy(python_theta)
        python_vz[i]=vz(python_theta) 
    
    #Finally deal with cell centred values, rho and T   

    python_density=np.zeros([len(python_r),len(python_theta)]) #We make an array including ghost cells
    python_T_e=np.zeros([len(python_r),len(python_theta)])

    python_density[0:-2,0:-2]=pluto_density #The ghost cells will retain a zero density - used as a flag in a bit
    python_T_e[0:-2,0:-2]=pluto_temperature



    #Now we need to turn these arrays into linear vectors for output

    ir=[]
    r=[]
    itheta=[]
    theta=[]
    inwind=[]
    v_x=[]
    v_y=[]
    v_z=[]
    density=[]
    T=[]



    for i in range(len(python_r)):
        for j in range(len(python_theta)):
            ir.append(i)
            r.append(python_r[i])
            itheta.append(j)
            theta.append(np.degrees(python_theta[j]))
            if python_density[i][j]==0.0:
                inwind.append(-1)   #these are ghost cells
            else:
                inwind.append(0)    #wind cells
            v_x.append(python_vx[i][j])
            v_y.append(python_vy[i][j])
            v_z.append(python_vz[i][j])
            density.append(python_density[i][j])
            T.append(python_T_e[i][j])
        



    titles=["ir","itheta","inwind","r","theta","v_x","v_y","v_z","density","T"]

    #This next line defines formats for the output variables. This is set in a dictionary
    fmt='%13.6e'
    fmts={'ir':'%03i',
        'itheta':'%03i',   
        'inwind':'%01i',         
        'r':fmt,
        'theta':fmt,
        'v_x':fmt,
        'v_y':fmt,
        'v_z':fmt,
        'density':fmt,
        'T':fmt}

    #set the filename 

    fname="%08d"%ifile+".pluto"
    out=open(fname,'w')
    
    #and output

    out_dat=Table([ir,itheta,inwind,r,theta,v_x,v_y,v_z,density,T],names=titles)
    ascii.write(out_dat,out,formats=fmts)
    out.close()
    return(fname)


            
#This makes a python input file - if the versionof python changes such that new parametets are required, this must be edited.    


def python_input_file(fname,data,cycles=2):
    output=open(fname+".pf",'w')
    output.write("System_type(star,binary,agn,previous)          "+data["system_type"]+"  \n")
    output.write("\n")
    output.write("### Parameters for the Central Object\n")
    output.write("Central_object.mass(msol)                  "+str(data["CENT_MASS"]/c.M_sun.cgs.value)+"\n")
    output.write("Central_object.radius(cm)                  "+str(data["CENT_RADIUS"])+"\n")
    output.write("\n")
    output.write("### Parameters for the Disk (if there is one)\n")
    output.write("\n")
    if data["disk_radiation"]=="yes":
        output.write("Disk.type(none,flat,vertically.extended)       flat\n")
        output.write("Disk.radiation(yes,no)      yes\n")
        output.write("Disk.temperature.profile(standard,readin,yso) standard\n") 
#        output.write("Disk.T_profile_file() max_40K_disk.dat \n")        #Sometimes we want to use a user define disk temp file.        
        output.write("Disk.rad_type_to_make_wind(bb,models) bb\n")        
        output.write("Disk.mdot(msol/yr) "+str(data["PY_DISK_MDOT"])+"\n")
        output.write("Disk.radmax(cm) "+str(data["DISK_TRUNC_RAD"])+"\n")
    else:
        output.write("Disk.radiation(yes,no)      no\n")        
    output.write("\n")
    output.write("### Parameters for BL or AGN\n")
    output.write("\n")
    if data["boundary_layer"]=="no":
        output.write("Boundary_layer.radiation(yes,no)                   no\n")
    else:
        output.write("Boundary_layer.radiation(yes,no)                   yes\n")        
        output.write("Boundary_layer.rad_type_to_make_wind(bb,models,power) bb\n")
        output.write("Boundary_layer.temp(K) "+str(data["T_BL"])+"\n")
        output.write("Boundary_layer.luminosity(ergs/s)  "+str(data["L_BL"])+"\n")        
    if data["cent_spectype"]=="brem":
        output.write("Central_object.radiation(yes,no)     yes\n")
        output.write("Central_object.rad_type_to_make_wind(bb,models,power,cloudy,brems)   brems\n")
        output.write("AGN.bremsstrahlung_temp(K) "+str(data["T_x"])+"\n")
        output.write("Central_object.luminosity(ergs/s) "+str(data["L_2_10"])+"\n")
        output.write("Central_object.geometry_for_source(sphere,lamp_post) sphere\n")
    elif data["cent_spectype"]=="bb":
        output.write("Central_object.radiation(yes,no)     yes\n")
        output.write("Central_object.rad_type_to_make_wind(bb,models,power,cloudy,brems)   bb\n")
        output.write("Central_object.blackbody_temp(K)                        "+str(data["T_star"])+"\n") 
        output.write("Central_object.geometry_for_source(sphere,lamp_post) sphere\n")   
    elif data["cent_spectype"]=="models":
        output.write("Central_object.radiation(yes,no)     yes\n")
        output.write("Central_object.rad_type_to_make_wind(bb,models,power,cloudy,brems)   models\n")
        output.write("Input_spectra.model_file          model.ls\n")
        output.write("Central_object.luminosity(ergs/s) "+str(data["L_2_10"])+"\n")
        output.write("Central_object.geometry_for_source(sphere,lamp_post) sphere\n")
    elif data["cent_spectype"]=="none":
        output.write("Central_object.radiation(yes,no)     no\n")
    output.write("\n")
    output.write("### Parameters descibing the various winds or coronae in the system\n")
    output.write("\n")
    if data["wind_radiation"]=="yes":
        output.write("Wind.radiation(yes,no) yes\n")
    else:
        output.write("Wind.radiation(yes,no) no\n")
    output.write("Wind.number_of_components  1\n")
    output.write("Wind.type(SV,star,hydro,corona,kwd,homologous,yso,shell,imported)  imported \n")
    output.write("Wind.coord_system(spherical,cylindrical,polar,cyl_var)  polar\n")
    output.write("Wind.dim.in.x_or_r.direction               30\n")
    output.write("Wind.dim.in.z_or_theta.direction           30\n")
    output.write("\n")
    output.write("### Parameters associated with photon number, cycles,ionization and radiative transfer options\n")
    output.write("\n")
    output.write("Photons_per_cycle        "+str(data["NPHOT"])+"\n")
    output.write("Ionization_cycles        "+str(cycles)+"\n")
    output.write("Spectrum_cycles          0\n")
    output.write("Wind.ionization(on.the.spot,ML93,LTE_tr,LTE_te,fixed,matrix_bb,matrix_pow)  matrix_pow\n")
    if data["line_trans"]=="macro":
        output.write("Line_transfer(pure_abs,pure_scat,sing_scat,escape_prob,thermal_trapping,macro_atoms,macro_atoms_thermal_trapping)   macro_atoms_escape_prob\n")
        output.write("Atomic_data  data/h10_hetop_standard80.dat\n")
        output.write("Matom_transition_mode(mc_jumps,matrix) matrix\n")
    elif data["line_trans"]=="simple":
        output.write("Line_transfer(pure_abs,pure_scat,sing_scat,escape_prob,thermal_trapping,macro_atoms,macro_atoms_thermal_trapping)   escape_prob\n")
        output.write("Atomic_data  data/standard80.dat\n")
    output.write("Surface.reflection.or.absorption(reflect,absorb,thermalized.rerad)    thermalized.rerad\n")
    output.write("Wind_heating.extra_processes(none,adiabatic,nonthermal,both)   none\n")
    output.write("\n")
    output.write("### Parameters for Domain 0\n")
    output.write("\n")
    output.write("Wind.model2import "+fname+".pluto\n")
    output.write("Wind.t.init                                10000\n")
    output.write("Wind.filling_factor(1=smooth,<1=clumped)   1\n")
    output.write("\n")
    output.write("### Parameters for Reverberation Modeling (if needed)\n")
    output.write("\n")    
    output.write("Reverb.type(none,photon,wind,matom)   none\n")
    output.write("\n")    
    output.write("### Other parameters\n")
    output.write("\n")    
    output.write("Photon_sampling.approach(T_star,cv,yso,AGN,min_max_freq,user_bands,cloudy_test,wide,logarithmic)  logarithmic\n")
    output.write("Photon_sampling.nbands                     10\n")
    output.write("Photon_sampling.low_energy_limit(eV)       0.13333\n")
    output.write("Photon_sampling.high_energy_limit(eV)      500\n")
    
    output.close()
    return
    
    


    
def pluto2py(ifile):

    D=pp.pload(ifile)

    # We need the definitions file - so we know the conversion factors.

    UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY=get_units('definitions.h')

    # Open an output file 

    fname="%08d"%ifile+".pluto"

    # Preamble

    out=open(fname,'w')
    out.write("# This is a file generated by hydro_to_python\n")
    out.write("# We can put any number of comments in behind # signs\n")
    out.write("# By default, the order of coordinates are \n")
    out.write("#                r, theta phi for spherical polars\n")
    out.write("#                         x,y,z        for carteisan\n")
    out.write("#                         or w, z, phi    for cylindrical\n")


    titles=[]
    titles=titles+["ir","r_cent","r_edge"]
    titles=titles+["itheta","theta_cent","theta_edge"]
    titles=titles+["v_r","v_theta","v_phi","density","temperature"]

    r_edge=[]
    r_ratio=(D.x1[2]-D.x1[1])/(D.x1[1]-D.x1[0])
    dr=(D.x1[1]-D.x1[0])/(0.5*(1.0+r_ratio))
    r_edge.append(D.x1[0]-0.5*dr)
    for i in range(len(D.x1)-1):
        r_edge.append(r_edge[-1]+dr)
        dr=dr*r_ratio
    
    
    r_edge=np.array(r_edge)    

    theta_edge=[]
    theta_ratio=(D.x2[2]-D.x2[1])/(D.x2[1]-D.x2[0])
    dtheta=(D.x2[1]-D.x2[0])/(0.5*(1.0+theta_ratio))
    theta_min=D.x2[0]-0.5*dtheta
    if theta_min<0.0:
        theta_min=0.0
    theta_edge.append(theta_min)
    for i in range(len(D.x2)-1):
        theta_edge.append(theta_edge[-1]+dtheta)
        dtheta=dtheta*theta_ratio
    if (theta_edge[-1]+(D.x2[-1]-theta_edge[-1])*2.0)>(np.pi/2.0):
        D.x2[-1]=(theta_edge[-1]+(np.pi/2.0))/2.0

    theta_edge=np.array(theta_edge)    

    col0=np.array([])
    col1=np.array([])
    col2=np.array([])
    col3=np.array([])
    col4=np.array([])
    col5=np.array([])
    col6=np.array([])
    col7=np.array([])
    col8=np.array([])
    col9=np.array([])
    col10=np.array([])

    fmt='%013.6e'

    #This next line defines formats for the output variables. This is set in a dictionary
    fmts={    'ir':'%03i',    
        'r_cent':fmt,
        'r_edge':fmt,
        'itheta':'%i',    
        'theta_cent':fmt,
        'theta_edge':fmt,
        'v_r':fmt,
        'v_theta':fmt,
        'v_phi':fmt,
        'density':fmt,
        'temperature':fmt}

    for j in range(len(D.x2)):
        col0=np.append(col0,np.arange(len(D.x1)))
        col1=np.append(col1,D.x1*UNIT_LENGTH)
        col2=np.append(col2,r_edge*UNIT_LENGTH)
        col3=np.append(col3,np.ones(len(D.x1))*j)
        col4=np.append(col4,np.ones(len(D.x1))*D.x2[j])
        col5=np.append(col5,np.ones(len(D.x1))*theta_edge[j])
        col6=np.append(col6,np.transpose(D.vx1)[j]*UNIT_VELOCITY)
        col7=np.append(col7,np.transpose(D.vx2)[j]*UNIT_VELOCITY)
        col8=np.append(col8,np.transpose(D.vx3)[j]*UNIT_VELOCITY)
        col9=np.append(col9,np.transpose(D.rho)[j]*UNIT_DENSITY)
        col10=np.append(col10,np.transpose(D.T)[j])

    out_dat=Table([col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10],names=titles)
    ascii.write(out_dat,out,formats=fmts)
    out.close()
    return
    
    
    
#This converts accelerations output by python into an accelerations file to be read in by pluto    

def accel_calc(ifile,radforce=0): 


    py_driving=ascii.read("%08d"%(ifile)+"_py_driving.dat")
        
    D=pp.pload(ifile)

    # We need the definitions file - so we know the conversion factors.

    UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY=get_units('definitions.h')
    UNIT_ACCELERATION=UNIT_VELOCITY*UNIT_VELOCITY/UNIT_LENGTH    

    # set up arrays to hold the data to output

    gx_es=[]
    gy_es=[]
    gz_es=[]
    
    gx_bf=[]
    gy_bf=[]
    gz_bf=[]
    
    gx=[]
    gy=[]
    gz=[]
    
    odd=0.0
       
    #We simply loop over all the lines in the fie
    
    for i in range(len(py_driving["rho"])):
        if (py_driving["rho"][i]/(D.rho[py_driving["i"][i]][py_driving["j"][i]]*UNIT_DENSITY))-1.>1e-6: #A test to make sure the geometry is the same
            odd=odd+1 #We dont really do anythin with this though!

        #Electron scattering acceleration - convert to the right units for pluto
        gx_es.append(py_driving["es_f_x"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))
        gy_es.append(py_driving["es_f_y"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))
        gz_es.append(py_driving["es_f_z"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))
        #BF scattering acceleration 
        gx_bf.append(py_driving["bf_f_x"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))
        gy_bf.append(py_driving["bf_f_y"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))
        gz_bf.append(py_driving["bf_f_z"][i]/(py_driving["rho"][i]*py_driving["vol"][i]))

        #These lines decide what we actually add into the acceleration - for line driving, electron scattering is included automatically by the M+1 term used in pluto - so we are only using bound free
        gx.append(gx_bf[-1])
        gy.append(0.0)    
        gz.append(gz_bf[-1])

    fmt='%013.6e'
        
    
    fmts2={'ir':'%03i',
        'rcent':fmt,
        'itheta':'%03i',
        'thetacent':fmt, 
        'rho':fmt,            
        'gx':fmt,
        'gy':fmt,
        'gz':fmt,
        }
        
    col0=py_driving["i"]
    col1=py_driving["rcen"]
    col2=py_driving["j"]
    col3=py_driving["thetacen"]
    col4=py_driving["rho"]


    titles=[]
    titles=titles+["ir","rcent","itheta","thetacent","rho"]
    titles=titles+["gx","gy","gz"]

    out=open("py_accelerations.dat",'w')
    out_dat=Table([col0,col1,col2,col3,col4,gx,gy,gz],names=titles)
    ascii.write(out_dat,out,formats=fmts2)
    out.close()    
    
    
    return(odd)

#This is the routine that is used to compute prefectors for the blondin heating and cooling rates - this is a little legacy - I think it still works OK, the idea is to take the last set of heating anbd cooling rates from the dbl file - copmare them to the heating and cooling rates in python, and export updated 'prefactors'. 


def pre_calc(ifile,radforce=0): 
    max_change=0.9
    max_accel_change=0.9

    heatcool=ascii.read("%08d"%(ifile)+"_py_heatcool.dat")
    D=pp.pload(ifile)

        
    # We need the definitions file - so we know the conversion factors.

    UNIT_DENSITY,UNIT_LENGTH,UNIT_VELOCITY=get_units('definitions.h')
    UNIT_ACCELERATION=UNIT_VELOCITY*UNIT_VELOCITY/UNIT_LENGTH    

    comp_h_pre=[]
    comp_c_pre=[]
    xray_h_pre=[]
    brem_c_pre=[]
    line_c_pre=[]
        
    odd=0.0
    itest=0
    
    for i in range(len(heatcool["rho"])):
        if (heatcool["rho"][i]/(D.rho[heatcool["i"][i]][heatcool["j"][i]]*UNIT_DENSITY))-1.>1e-6:
            odd=odd+1
        nenh=D.ne[heatcool["i"][i]][heatcool["j"][i]]*D.nh[heatcool["i"][i]][heatcool["j"][i]]
        nhnh=D.nh[heatcool["i"][i]][heatcool["j"][i]]*D.nh[heatcool["i"][i]][heatcool["j"][i]]
        
        ideal_prefactor=(heatcool["heat_comp"][i]/(D.ch[heatcool["i"][i]][heatcool["j"][i]]*nenh))
        change=ideal_prefactor/D.ch_pre[heatcool["i"][i]][heatcool["j"][i]]
        if change<max_change:
            change=max_change
        elif change>(1./max_change):
            change=(1./max_change)
        comp_h_pre.append(change*D.ch_pre[heatcool["i"][i]][heatcool["j"][i]])
            
        ideal_prefactor=(heatcool["cool_comp"][i]/(D.cc[heatcool["i"][i]][heatcool["j"][i]]*nenh))
        change=ideal_prefactor/D.cc_pre[heatcool["i"][i]][heatcool["j"][i]]
        if change<max_change:
            change=max_change
        elif change>(1./max_change):
            change=(1./max_change)
        comp_c_pre.append(change*D.cc_pre[heatcool["i"][i]][heatcool["j"][i]])
    
        ideal_prefactor=(heatcool["cool_lines"][i]/(D.lc[heatcool["i"][i]][heatcool["j"][i]]*nenh))
        change=ideal_prefactor/D.lc_pre[heatcool["i"][i]][heatcool["j"][i]]
        if change<max_change:
            change=max_change
        elif change>(1./max_change):
            change=(1./max_change)
        line_c_pre.append(change*D.lc_pre[heatcool["i"][i]][heatcool["j"][i]])
        
        ideal_prefactor=(heatcool["cool_ff"][i]/(D.bc[heatcool["i"][i]][heatcool["j"][i]]*nenh))
        change=ideal_prefactor/D.bc_pre[heatcool["i"][i]][heatcool["j"][i]]
        if change<max_change:
            change=max_change
        elif change>(1./max_change):
            change=(1./max_change)
        brem_c_pre.append(change*D.bc_pre[heatcool["i"][i]][heatcool["j"][i]])
    
        ideal_prefactor=(heatcool["heat_xray"][i]/(D.xh[heatcool["i"][i]][heatcool["j"][i]]*nhnh))
        change=ideal_prefactor/D.xh_pre[heatcool["i"][i]][heatcool["j"][i]]
        if change<max_change:
            change=max_change
        elif change>(1./max_change):
            change=(1./max_change)
        xray_h_pre.append(change*D.xh_pre[heatcool["i"][i]][heatcool["j"][i]])
        
            
    fmt='%013.6e'

    #This next line defines formats for the output variables. This is set in a dictionary
    fmts={    'ir':'%03i',
        'rcent':fmt,
        'itheta':'%03i',
        'thetacent':fmt,    
        'rho':fmt,
        'comp_h_pre':fmt,
        'comp_c_pre':fmt,
        'xray_h_pre':fmt,
        'line_c_pre':fmt,
        'brem_c_pre':fmt,
        'gx':fmt,
        'gy':fmt,
        'gz':fmt,
        }  
          
    titles=[]
    titles=titles+["ir","rcent","itheta","thetacent","rho"]
    titles=titles+["comp_h_pre","comp_c_pre","xray_h_pre","brem_c_pre","line_c_pre"]
    titles=titles+["gx","gy","gz"]    
    
    col0=heatcool["i"]
    col1=heatcool["rcen"]
    col2=heatcool["j"]
    col3=heatcool["thetacen"]
    col4=heatcool["rho"]
    col5=comp_h_pre
    col6=comp_c_pre
    col7=xray_h_pre
    col8=brem_c_pre
    col9=line_c_pre
    col10=gx
    col11=gy
    col12=gz

    out=open("prefactors.dat",'w')

    out_dat=Table([col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12],names=titles)
    ascii.write(out_dat,out,formats=fmts)
    out.close()
    
    if radforce:
        fmts2={'ir':'%03i',
            'rcent':fmt,
            'itheta':'%03i',
            'thetacent':fmt,    
            'gx_es':fmt,
            'gy_es':fmt,
            'gz_es':fmt,
            'gx_bf':fmt,
            'gx_bf':fmt,
            'gx_bf':fmt,
            'gx_line':fmt,
            'gx_line':fmt,
            'gx_line':fmt,
            }
    
        titles=[]
        titles=titles+["ir","rcent","itheta","thetacent"]
        titles=titles+["gx_es","gy_es","gz_es"]
        titles=titles+["gx_bf","gy_bf","gz_bf"]
        titles=titles+["gx_line","gy_line","gz_line"]
    
        out=open("accelerations.dat",'w')
        out_dat=Table([col0,col1,col2,col3,gx_es,gy_es,gz_es,gx_bf,gy_bf,gz_bf,gx_line,gy_line,gz_line],names=titles)
        ascii.write(out_dat,out,formats=fmts2)
        out.close()    
    
    
    return(odd)
        
