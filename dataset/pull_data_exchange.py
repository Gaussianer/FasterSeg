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
    use_only_synthetic_data=True #if False: use only real data for testing

    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    
    #clone the data from data_exchange repo
    clone("KHeap25", "data_exchange", datasetPath)

    #generate lists with the filenames 
    if use_only_synthetic_data==True:
        annotationFileList=generateFilenamesList(datasetPath,"/data_exchange/synthetic_data/annotations","_polygons.json")
        rawFileList=generateFilenamesList(datasetPath,"/data_exchange/synthetic_data/original_images","_raw.png")

        #split it up
        train_per = 60
        val_per = 20
        test_per = 20
        num_raw = countFiles(datasetPath+"/data_exchange/synthetic_data/original_images")

        num_train = round(num_raw*(train_per/100), 0)
        num_val = round(num_raw*(val_per/100), 0)
        num_test = round(num_raw*(test_per/100), 0)

        moveFiles(num_train, num_val, num_test, "original_images", datasetPath, rawFileList)
        moveFiles(num_train, num_val, num_test, "annotations", datasetPath, annotationFileList)
    
    else:
        annotationFileList=generateFilenamesList(datasetPath,"/data_exchange/synthetic_data/annotations","_polygons.json")
        rawFileList=generateFilenamesList(datasetPath,"/data_exchange/synthetic_data/original_images","_raw.png")

        realAnnotationFileList=generateFilenamesList(datasetPath,"/data_exchange/real_data/annotations","_polygons.json")
        realRawFileList=generateFilenamesList(datasetPath,"/data_exchange/real_data/original_images","_raw.png")

        #split it up
        train_per = 80
        val_per = 20
        test_per = 0 #use only real data for testing
        num_raw = countFiles(datasetPath+"/data_exchange/synthetic_data/original_images")

        num_train = round(num_raw*(train_per/100), 0)
        num_val = round(num_raw*(val_per/100), 0)
        num_test = round(num_raw*(test_per/100), 0)

        moveFiles(num_train, num_val, num_test, "original_images", datasetPath, rawFileList, realRawFileList)
        moveFiles(num_train, num_val, num_test, "annotations", datasetPath, annotationFileList, realAnnotationFileList)

        movedRealAnnotationFileList=generateFilenamesList(datasetPath, "/annotations/test","_polygons.json")
        movedRealRawFileList=generateFilenamesList(datasetPath, "/original_images/test","_raw.png")

        renameFilesInList(movedRealAnnotationFileList, "real_")
        renameFilesInList(movedRealRawFileList, "real_")



def renameFilesInList(fileList, prefix):
    
    for file in fileList:

        splittedPath=file.split('/')
        
        newName=prefix+splittedPath[-1]
        splittedPath.pop()
        splittedPath.append(newName)
        
        os.rename(file, "/".join(splittedPath)) 


def generateFilenamesList(dataset_path, sub_dir, file_suffix):
    search_files = os.path.join(dataset_path+sub_dir,"*"+file_suffix) 
    filelist = glob.glob(search_files)
    filelist.sort()
    return filelist

def moveFiles(num_train, num_val, num_test, prefix_directory, dataset_path, filenames_list, real_data_filenames_list=False):
    
    count = 0
    if real_data_filenames_list==False:
        
        for f in filenames_list:
            if count < num_train:
                moveFile(f, dataset_path+"/"+prefix_directory+"/train")
            elif count < num_train+num_val:
                moveFile(f,dataset_path+"/"+prefix_directory+"/val")
            elif count < num_train+num_val+num_test:
                moveFile(f, dataset_path+"/"+prefix_directory+"/test")
            
            count = count +1
    else:
        for f in filenames_list:
            if count < num_train:
                moveFile(f, dataset_path+"/"+prefix_directory+"/train")
            elif count < num_train+num_val:
                moveFile(f,dataset_path+"/"+prefix_directory+"/val")
            count = count +1

        for f in real_data_filenames_list:
            moveFile(f, dataset_path+"/"+prefix_directory+"/test")
          
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