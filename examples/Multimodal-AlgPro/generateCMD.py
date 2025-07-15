# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:35:51 2022

@author: tmp
"""

import os, sys
from itertools import combinations
positiveSeqFile = './examples/Multimodal-AlgPro/data/Independent/trpos2840.txt'
negativeSeqFile = './examples/Multimodal-AlgPro/data/Independent/trneg2840.txt'

# positiveFeaFiles = []
# negativeFeaFiles = []
# dataPath = './data'
# for f in os.listdir(dataPath):
#     if f.startswith('po'):
#         positiveFeaFiles.append('%s/%s' %(dataPath,f))
#     elif f.startswith('ne'):
#         negativeFeaFiles.append('%s/%s' %(dataPath,f))

    
positiveFeaFiles = ['./examples/Multimodal-AlgPro/data/Independent/trpo-aac_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trpo-DFMCA_PSSM.txt',
                    './examples/Multimodal-AlgPro/data/Independent/trpo-dpc_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trpo-DP_PSSM.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trpo-pse_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trpo-pssm400.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trpo-pssm_ac.txt', './examples/Multimodal-AlgPro/data/Independent/trpo-single_Average.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trpo-AAC.txt','./examples/Multimodal-AlgPro/data/Independent/trpo-DC.txt','./examples/Multimodal-AlgPro/data/Independent/trpo-CTD.txt']

negativeFeaFiles = ['./examples/Multimodal-AlgPro/data/Independent/trne-aac_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trne-DFMCA_PSSM.txt',
                    './examples/Multimodal-AlgPro/data/Independent/trne-dpc_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trne-DP_PSSM.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trne-pse_pssm.txt', './examples/Multimodal-AlgPro/data/Independent/trne-pssm400.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trne-pssm_ac.txt', './examples/Multimodal-AlgPro/data/Independent/trne-single_Average.txt', 
                    './examples/Multimodal-AlgPro/data/Independent/trne-AAC.txt','./examples/Multimodal-AlgPro/data/Independent/trne-DC.txt','./examples/Multimodal-AlgPro/data/Independent/trne-CTD.txt']

# spcLenList = []

feaDict = {}
for i in range(len(positiveFeaFiles)):
    posFile = positiveFeaFiles[i]
    negFile = negativeFeaFiles[i]
    feaName = os.path.split(posFile)[-1].split('-')[-1].split('.')[0]
    modelName = './examples/Multimodal-AlgPro/model/Multimodal/%s.py' %feaName
    # assert os.path.exists(modelName)
    feaDict[feaName] = (posFile,negFile,modelName,200)

repeatTime = 5

cmdTemp = 'python running.py --dataType protein %s --dataEncodingType onehot %s --dataTrainFilePaths %s --dataTrainLabel 1 0%s --dataSplitScale 0.8 --modelLoadFile examples/Multimodal-AlgPro/model/Multimodal/CNN.py %s --verbose 1 --showFig 0 --outSaveFolderPath %s --savePrediction 1 --saveFig 1 --batch_size 256 --epochs 20 --shuffleDataTrain 1 --spcLen 1000 %s --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0%s'
errCMDList = []
feaNames = list(feaDict.keys())

for repeatNum in range(repeatTime):
    for combNum in range(11):
        combIterObj = combinations(feaNames, combNum + 1)
        
        for combIter in combIterObj:
            dataType = ''
            dataEncodingType = ''
            dataTrainFilePaths = positiveSeqFile+' '+negativeSeqFile #oriFile needed
            dataTrainLabel = ''
            modelLoadFile = ''
            outSaveFolderPath = 'outs%d/' %repeatNum
            spcLen = ''
            dataTrainModelInd = ''
            modelCount = 1
            
            for feaName in combIter:
                _posFile,_negFile,_modelName,_spcLen = feaDict[feaName]
                
                dataType += ' other'
                dataEncodingType += ' other'
                dataTrainFilePaths += ' ' + _posFile + ' ' + _negFile
                dataTrainLabel += ' 1 0'
                modelLoadFile += ' ' + _modelName
                outSaveFolderPath += feaName + '__'
                spcLen += ' %d' %_spcLen
                dataTrainModelInd += ' %d %d' %(modelCount,modelCount)
                modelCount += 1
            outSaveFolderPath = outSaveFolderPath[:-2]
            cmd = cmdTemp %(dataType, dataEncodingType, dataTrainFilePaths, dataTrainLabel, modelLoadFile, outSaveFolderPath, spcLen, dataTrainModelInd)
            print('#' * 10)
            print(cmd)
            isErr = os.system(cmd)
            if isErr:
                errCMDList.append('%d::%s' %(repeatNum,cmd))
print('#'*50)
print('err:')
for cmd in errCMDList:
    print('*'*10)
    print(cmd)
