import os
import numpy as np
import csv
import pandas as pd

import SimpleITK as sitk
import torch

import nrrd
import radiomics
from radiomics import featureextractor, getTestCase
extractor = featureextractor.RadiomicsFeatureExtractor()

print("radiomics version:", radiomics.__version__)


# save nrrd
def pt2nrrd(pid:str):

    img_select = ['pet', 'rtct', 'rtst_data', 'rtdose']
    
    for SEL in img_select:
        pt_path = f"./DB/tensor/{SEL}/{pid}.pt"
        img = torch.load(pt_path).numpy()
        nrrd.write(f"./DB/nrrd/{SEL}/{pid}.nrrd", img)


# completed
"""

for i in os.listdir("./DB/tensor/rtst_data"):
    pid = i.split(".pt")[0]

    pt2nrrd(pid)
"""


# extract nrrd
def extractnrrd(imagePath:str, maskPath:str):
    columns = ["pid"]

    result = extractor.execute(imagePath, maskPath)

    # generate columns
    for kk, _ in result.items():
        columns.append(kk)

    
    df = pd.DataFrame(columns=columns)

    # temp
    df = df.append(result, ignore_index=True)
    df["pid"] = maskPath.split(".nrrd")[0][-8:]

    return df



# repeat
def selectFolder(targetImg:str, saveFolder:str):
    # targetImg: ['pet', 'rtct', 'rtdose', 'rtst_data']

    # folder setting
    initFolder = f"./DB/nrrd/"
    taskFolder = os.path.join(initFolder, f"{targetImg}")
    maskFolder = os.path.join(initFolder, f"rtst_data")

    taskList = os.listdir(taskFolder)
    maskList = os.listdir(maskFolder)


    totalColumns = ["pid"]

    # sample result for creating columns
    result = extractor.execute('nrrd sample file(ex pet.nrrd)', 'nrrd sample file(ex rtst.nrrd)')

    for kk, _ in result.items():
        totalColumns.append(kk)

    df = pd.DataFrame(columns=totalColumns)

    if maskList == taskList:
        # maskPath = os.path.join(mask, maskList)
        # taskPath = os.path.join(taskFolder, maskList)
        
        for l in maskList:
            taskPath = os.path.join(taskFolder, l)
            maskPath = os.path.join(maskFolder, l)
            
            
            output = extractnrrd(taskPath, maskPath)
            df = df.append(output, ignore_index=True)
            
    
    else:
        print("Masklist and Tasklist components are not the same")

    df.to_csv(f'{saveFolder}/{targetImg}.csv')
    
    return df


# execution code

selectFolder("pet", "SAVE_PATH(folder)")
selectFolder("rtct", "SAVE_PATH(folder)")
selectFolder("rtdose", "SAVE_PATH(folder)")


