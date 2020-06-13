#This script generates the "*_mapping_list.txt" file.
#That is a file where the names of the data pairs are stored. 
#A pair consists of the raw picture and the corresponding annotated picture.
#The mapping list is needed for the training of the FasterSeg network.
#
#Changes in compartion to create_mapping_list.py
#-select train, val, test at the beginning
#-reduce the number of for loop iterations in func createMappingList()
#-cut datasetPath from the string which is written in the lists
#-add func createTrainValMappingList()
#
# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
import re, time

#Func: main()################################################
#This is the mainfunction of the script
#  
#############################################################
def main():
    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    
    # how to search for the pathnames
    # all *_labelTrainIds.png files in the annotations folder
    # all *_raw images in the original_images folder
    # separated by train, val and test 
    
    os.chdir(datasetPath)
    search_annotations_train = os.path.join( "annotations/train" , "*_labelTrainIds.png" ) #after testing search directly for _labelTrainIds.png
    search_annotations_val = os.path.join( "annotations/val" , "*_labelTrainIds.png" ) 
    search_annotations_test = os.path.join( "annotations/test" , "*_labelTrainIds.png" )

    search_raw_train = os.path.join( "original_images/train" , "*_raw.png" )
    search_raw_val = os.path.join( "original_images/val" , "*_raw.png" )
    search_raw_test = os.path.join( "original_images/test" , "*_raw.png" )

    # search and sort the lists with the file names

    filesAnnotationsTrain = glob.glob(search_annotations_train)
    filesAnnotationsTrain.sort()
    filesAnnotationsVal = glob.glob(search_annotations_val)
    filesAnnotationsVal.sort()
    filesAnnotationsTest = glob.glob(search_annotations_test)
    filesAnnotationsTest.sort()

    filesRawTrain = glob.glob(search_raw_train)
    filesRawTrain.sort()
    filesRawVal = glob.glob(search_raw_val)
    filesRawVal.sort()
    filesRawTest = glob.glob(search_raw_test)
    filesRawTest.sort()

    #parse filesRaw and filesAnnotation and store the names pairwise in the corresponding text file 
    #first for training dataset
    try:
        createMappingList(filesRawTrain, filesAnnotationsTrain, "train", datasetPath)
    except:
        print("error: creation of train_mapping_list.txt failed")

    #second for validation dataset
    try:
        createMappingList(filesRawVal, filesAnnotationsVal, "val", datasetPath)
    except:
        print("error: creation of val_mapping_list.txt failed")

    #third for test dataset
    try:
        createMappingList(filesRawTest, filesAnnotationsTest, "test", datasetPath)
    except:
        print("error: creation of test_mapping_list.txt failed")

    #create the train_val_mapping_list.txt file
    try:
        createTrainValMappingList(filesRawTrain, filesRawVal, filesAnnotationsTrain, filesAnnotationsVal, datasetPath)
    except:
        print("error: creation of train_val_mapping_list.txt failed")

#Func: createMappingList()################################################
#This function parses the raw list with attention to dataset type.
#It creates the mapping_file for train, val or test dataset and stores the 
#pair of raw file name and coresponding annotation file name in ecach line 
#of the mapping_list file.
#Params:
#raw: list of raw data file names
#annotation: list of annotation file names
#dataset: type of the dataset (train, val or test)
#datasetPath: path of the dataset folder (from env variable) 
##########################################################################
def createMappingList(raw, annotation, dataset, datasetPath):

    if annotation != [] and raw != []:
        if dataset=="train":
            mapping_file_path = datasetPath+"/train_mapping_list.txt"
            mapping_file = open(mapping_file_path,'w')

            for i in raw:
                if "train" in i:
                    mapping_file.write(i+" ")
                    search_partner = getNameFromPath(i, "raw.png")

                    for j in annotation:
                        if "train" in j:
                            if search_partner in j:
                                mapping_file.write(j+"\n")
                                break
                    
            mapping_file.close()    

        
        elif dataset=="val":
            mapping_file_path = datasetPath+"/val_mapping_list.txt"
            mapping_file = open(mapping_file_path,'w')

            for i in raw:
                if "val" in i:
                    mapping_file.write(i+" ")
                    search_partner = getNameFromPath(i, "raw.png")

                    for j in annotation:
                        if "val" in j:
                            if search_partner in j:
                                mapping_file.write(j+"\n")
                                break   
            mapping_file.close


        elif dataset=="test":
            mapping_file_path = datasetPath+"/test_mapping_list.txt"
            mapping_file = open(mapping_file_path,'w')

            for i in raw:
                if "test" in i:
                    mapping_file.write(i+" ")
                    search_partner = getNameFromPath(i, "raw.png")

                    for j in annotation:
                        if "test" in j:
                            if search_partner in j:
                                mapping_file.write(j+"\n")
                                break
            mapping_file.close
    else:
        print("no raw or annotation files for "+dataset)

#Func: getNameFromPath()################################################
#This function extract the name of the dataset from the path. For
#example route_00000_000001. This is neede to find the pairs of labeled
#image and raw image depending on the name.
#Params:
#dataset_path: path of the file
#format: format of the file, which name should be extracted
#        possible values: raw.png, labelTrainIds.png, polygons.json 
##########################################################################
def getNameFromPath(dataset_path, format):
    name = re.sub('_'+format, '', dataset_path)
    cut = name.find("route")
    name=name[cut:]
    return name

#Func: createTrainValMappingList()########################################
#This function generates the train_val_mapping_list.txt
#The pairs of train data names and val data names are stored in this list.
#
#Params: 
#train_raw: list of train raw data file names
#val_raw: list of val raw data file names
#train_annot: list of train annotation data file names
#val_annot: list of val annotation data file names
#datasetPath: path of the dataset folder (from env variable) 
##########################################################################
def createTrainValMappingList(train_raw, val_raw, train_annot, val_annot, datasetPath):
    
    if train_raw!= [] and val_raw != [] and train_annot != [] and  val_annot != []:

        mapping_file_path = datasetPath+"/train_val_mapping_list.txt"
        mapping_file = open(mapping_file_path,'w')

        for i in train_raw:
            if "train" in i:
                mapping_file.write(i+" ")
                search_partner = getNameFromPath(i, "raw.png")

                for j in train_annot:
                    if "train" in j:
                        if search_partner in j:
                            mapping_file.write(j+"\n")
                            break
        
        for i in val_raw:
            if "val" in i:
                mapping_file.write(i+" ")
                search_partner = getNameFromPath(i, "raw.png")

                for j in val_annot:
                    if "val" in j:
                        if search_partner in j:
                            mapping_file.write(j+"\n")
                            break
        mapping_file.close                

# call the main
if __name__ == "__main__":
    start=time.time()
    main()
    print("elapsed time:", time.time()-start)