# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:02:34 2019

@author: Amar Shivaram
"""
import os
from configparser import ConfigParser
from driver_program import driver_function
import warnings
warnings.filterwarnings("ignore")



def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Fetch the properties and configurations 
    prop = ConfigParser()
    
    #Reading the configuration file
    
    prop.read("properties.ini")    
    
#    print(prop['WorkingDirectory']['local_filepath'])
    directory = prop['WorkingDirectory']['local_filepath']
#    print(directory)
    os.chdir(directory)  ##### TO SET THE WORKING DIRECTORY

    
    #Calling the driver function
    
    driver_function(prop)

    
    
    
if __name__ == '__main__':
    main()