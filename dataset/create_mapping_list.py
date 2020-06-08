#This script generates the "*_mapping_list.txt" file.
#That is a file where the names of the data pairs are stored. 
#A pair consists of the raw picture and the corresponding annotated picture.
#The mapping list is needed for the training of the FasterSeg network.
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
import re, time

#Func: main()################################################
#This is the mainfunction of the script
# 
#  
#############################################################
def main():
    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    
    # how to search for all ground truth
    # all *_polygon.json files in the annotations folder
    # all *_raw images in the original_images folder
    
    search_annotations = os.path.join( datasetPath , "annotations" , "*" ,  "*_polygons.json" ) #after testing search directly for _labelTrainIds.png
    search_raw = os.path.join( datasetPath , "original_images" , "*" ,  "*_raw.png" ) #after testing change to _raw.png

    # search and sort the lists with the file names
    filesAnnotations = glob.glob(search_annotations)
    filesAnnotations.sort()
    filesRaw = glob.glob(search_raw)
    filesRaw.sort()

    #for testing change the filename of the annotations from "_polygons.json" to "_labelTrainIds.png"
    #finally select the _labelTrainIds.png directly, so the for loop is not necessary
    i=0
    for f in filesAnnotations:
        # create the output filename
        filesAnnotations[i] = f.replace( "_polygons.json" , "_labelTrainIds.png" ) #target name depends on the encoding method in json2labelImg()
        i=i+1


    #parse filesRaw and filesAnnotation and store the names pairwise in the corresponding text file 
    #first for training dataset
    try:
        writeNamesInFile(filesRaw, filesAnnotations, "train", datasetPath)
    except:
        print("error: creation of train_mapping_list failed")

    #second for validation dataset
    try:
        writeNamesInFile(filesRaw, filesAnnotations, "val", datasetPath)
    except:
        print("error: creation of val_mapping_list failed")

    #third for test dataset
    try:
        writeNamesInFile(filesRaw, filesAnnotations, "test", datasetPath)
    except:
        print("error: creation of test_mapping_list failed")


#Func: writeNamesInFile()################################################
#This function parses the annotation list with attention to dataset.
#It creates the mapping_file and stores the pair of raw file name 
#and annotation file name in ecach line of the mapping file.
#Params:
#raw: list of raw data file names
#annotation: list of annotation file names
#dataset: type of the dataset (train, val or test)
#datasetPath: path of the dataset folder (from env variable) 
##########################################################################
def writeNamesInFile(raw, annotation, dataset, datasetPath):

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


# call the main
if __name__ == "__main__":
    start=time.time()
    main()
    print("elapsed time:", time.time()-start)