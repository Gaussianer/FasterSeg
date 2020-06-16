#This script cleans up the dataset directories.
#It deletes the content of the train val and test
#sub directories and the whole data_exchange folder.
#This action keeps the repo clean and creates space
#for the following dataset's.


# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
import time, shutil, re

def main():
    
    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    # remove the annotation files
    os.system('rm -r ' + datasetPath + '/annotations/train/*')
    os.system('rm -r ' + datasetPath + '/annotations/val/*')
    os.system('rm -r ' + datasetPath + '/annotations/test/*')

    # remove the original images files
    os.system('rm -r ' + datasetPath + '/original_images/train/*')
    os.system('rm -r ' + datasetPath + '/original_images/val/*')
    os.system('rm -r ' + datasetPath + '/original_images/test/*')
    
    #remove the data_exchange folder
    os.system('rm -r -f ' + datasetPath + '/data_exchange')


# call the main
if __name__ == "__main__":
    start=time.time()
    main()
    print("elapsed time:", time.time()-start)