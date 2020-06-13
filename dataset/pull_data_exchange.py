#This script is used for the data exchange.
#It pulls the actual dataset from the git repository.
#We use this repository as a data exchange platform. 
#The reason for that is, to have a plublic place, which 
#we can access from everywhere. For example from the 
#Docker Container on the HPC-Server, from the ground truth 
#labeler, etc. If the datasets grow to more than one Gigabyte, 
#we have to find another place instead of github. Furthermore
#the script splits up the dataset into the three subsets of 
#train, val and test. The subsets will be moved to the right
#directories.
#Actual Version only works with same number of files in the
#data_exchange/annotations and data_exchange/original_images
#folders.
#
#Further points: 
#compare num_raw and num_poly to verify that the two folders
#has the same amount of files.


# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
import time, shutil, re

#Func: main()################################################
#This is the mainfunction of the script
#############################################################
def main():

    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    
    #clone the data from data_exchange repo
    clone("KHeap25", "data_exchange", datasetPath)

    #generate lists with the filenames 
    search_annotations = os.path.join( datasetPath+ "/data_exchange/annotations" , "*_polygons.json" ) 
    filesAnnotations = glob.glob(search_annotations)
    filesAnnotations.sort()

    search_raw = os.path.join( datasetPath+ "/data_exchange/original_images" , "*_raw.png" ) 
    filesRaw = glob.glob(search_raw)
    filesRaw.sort()

    #split it up
    train_per = 60
    val_per = 20
    test_per = 20

    num_raw = countFiles(datasetPath+"/data_exchange/original_images")
    #num_poly = countFiles(datasetPath+"/data_exchange/annotations")

    num_train = round(num_raw*(train_per/100), 0)
    num_val = round(num_raw*(val_per/100), 0)
    num_test = round(num_raw*(test_per/100), 0)

    # move the raw files (_raw.png) to the right directory
    count = 0
    for f in filesRaw:
        if count < num_train:
            moveFile(f, datasetPath+"/original_images/train")
        elif count < num_train+num_val:
            moveFile(f, datasetPath+"/original_images/val")
        elif count < num_train+num_val+num_test:
            moveFile(f, datasetPath+"/original_images/test")
        
        count = count +1

    #move the annotation files (polygons.json) to the right directory
    count = 0
    for f in filesAnnotations:
        if count < num_train:
            moveFile(f, datasetPath+"/annotations/train")
        elif count < num_train+num_val:
            moveFile(f, datasetPath+"/annotations/val")
        elif count < num_train+num_val+num_test:
            moveFile(f, datasetPath+"/annotations/test")
        
        count = count +1

    
#Func: clone()################################################
#This function clones a github repository into a specified
#location.
#params:
#git_user:  name of the git user
#repo: name of the repository
#location:  place where you want to store
##############################################################
def clone(git_user, repo, location):
    try:
        os.chdir(location)
        os.system('git clone https://github.com/'+ git_user + "/"+ repo + ".git")
        
    except:
        print("git clone faild")

#Func: countFiles(dir)########################################
#This function returns the number of files in a directory.
#It counts only files, no sub directories
#params:
#dir: path of the directory (string)
##############################################################
def countFiles(dir):
    onlyfiles = next(os.walk(dir))[2]
    return len(onlyfiles)

#Func: moveFile(file, destination)#############################
#Moves a file to the destination
#Params:
#file: whole path to the file which you want to move (string)
#destination: destination path  (string)
#
def moveFile(file, destination):
    
    name = getNameFromPath(file)
    if not os.path.exists(destination+'/'+name):
        shutil.move(file, destination)

#Func: getNameFromPath()########################################
#This function finds the name of the file in path and returns 
#it as a string. For example "route_000_000_raw.png"
#Params:
#dataset_path: path of the file
################################################################
def getNameFromPath(data_name):
    cut = data_name.find("route")
    data_name=data_name[cut:]
    return data_name


# call the main
if __name__ == "__main__":
    start=time.time()
    main()
    print("elapsed time:", time.time()-start)