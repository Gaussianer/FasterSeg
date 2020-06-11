############################################################################################
#This file provides helper functions for various scripts for training the FasterSeg model. 
#___________________________________________________________________________________________

# python imports
import os, re
from glob import glob

def GetTheNumberOfTrainingExamplesFor(subfolder):
    # Save current path, because we change the path
    savedPath = os.getcwd()

    # Where to look for the datasets
    datasetPath = os.environ['DATASET_PATH']
    specificAnnotationDirectoryPath = os.path.join(datasetPath, "annotations", subfolder)
    specificRawDirectoryPath = os.path.join(datasetPath, "original_images", subfolder)

    # File types we are looking for
    annotationFileTyp = "json"
    rawFileTyp = "png"

    listOfTheNumberOfFiles = []
    # Get the number of annotation data
    os.chdir(specificAnnotationDirectoryPath)
    listOfTheNumberOfFiles.append(len(glob('*.' + annotationFileTyp)))

    # Get the number of raw data
    os.chdir(specificRawDirectoryPath)
    listOfTheNumberOfFiles.append(len(glob('*.' + rawFileTyp)))

    # Go to the path saved at the beginning
    os.chdir(savedPath)

    return min(listOfTheNumberOfFiles)

if __name__ == "__main__" and __package__ is None:
    __package__ = "expected.package.name"