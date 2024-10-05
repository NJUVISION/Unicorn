# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, time
import numpy as np 
import subprocess
rootdir_tmc13 = os.path.split(__file__)[0]


def get_points_number(filedir):
    plyfile = open(filedir)
    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])
    
    return number

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item)
        except ValueError: continue
        
    return number



def gpcc_encode(filedir, bin_dir, posQuantscale=1, transformType=0, qp=22, 
                    cfgdir=None, test_time=False, show=True, version=21, attribute='color', angularEnabled=0):
    """
    """
    config =' --trisoupNodeSizeLog2=0' + \
            ' --neighbourAvailBoundaryLog2=8' + \
            ' --intra_pred_max_node_size_log2=6' + \
            ' --maxNumQtBtBeforeOt=4' + \
            ' --planarEnabled=1' + \
            ' --planarModeIdcmUse=0' + \
            ' --minQtbtSizeLog2=0' + \
            ' --positionQuantizationScale='+str(posQuantscale)
    # lossless
    if posQuantscale == 1: config += ' --mergeDuplicatedPoints=0' + \
                                ' --inferredDirectCodingMode=1'
    else: config += ' --mergeDuplicatedPoints=1'
    if qp is not None: 
        if version >=14:
            if transformType==0:
                # RAHT
                config+=' --convertPlyColourspace=1' + \
                        ' --transformType=0' + \
                        ' --qp='+str(qp) + \
                        ' --qpChromaOffset=0'+ \
                        ' --bitdepth=8'+ \
                        ' --attrOffset=0'+ \
                        ' --attrScale=1'+ \
                        ' --attribute='+attribute
            if transformType==1:
                # predict (for lossless coding)
                config+=' --convertPlyColourspace=1' + \
                        ' --transformType=1' + \
                        ' --numberOfNearestNeighborsInPrediction=3' + \
                        ' --levelOfDetailCount=12' + \
                        ' --intraLodPredictionSkipLayers=0' + \
                        ' --interComponentPredictionEnabled=0' + \
                        ' --adaptivePredictionThreshold=64' + \
                        ' --qp=4' + \
                        ' --qpChromaOffset=0' + \
                        ' --bitdepth=8' + \
                        ' --colourMatrix=8' + \
                        ' --attrOffset=0' + \
                        ' --attrScale=1' + \
                        ' --attribute='+attribute
                if attribute=='reflectance':
                    cfgdir = os.path.join(rootdir_tmc13, 'gpcc_cfg/ford.cfg')

        else:
            if transformType==0:
                # RAHT
                if version==13:
                    config+=' --convertPlyColourspace=1' + \
                            ' --transformType=1' + \
                            ' --qp='+str(qp) + \
                            ' --qpChromaOffset=1'+ \
                            ' --bitdepth=8'+ \
                            ' --attribute='+attribute
                if version==12:
                    config+=' --convertPlyColourspace=1' + \
                            ' --transformType=0' + \
                            ' --qp='+str(qp) + \
                            ' --qpChromaOffset=0'+ \
                            ' --bitdepth=8'+ \
                            ' --attrOffset=0'+ \
                            ' --attrScale=1'+ \
                            ' --attribute='+attribute

        if transformType==2:
            # predict & lifting
            config+=' --convertPlyColourspace=1' + \
                    ' --transformType=2' + \
                    ' --numberOfNearestNeighborsInPrediction=3' + \
                    ' --levelOfDetailCount=12' + \
                    ' --lodDecimator=0' + \
                    ' --adaptivePredictionThreshold=64' + \
                    ' --qp='+str(qp) + \
                    ' --qpChromaOffset=0'+ \
                    ' --bitdepth=8'+ \
                    ' --attrOffset=0'+ \
                    ' --attrScale=1'+ \
                    ' --attribute='+attribute

    if cfgdir is not None:
        config = ' --config='+cfgdir

    if angularEnabled:
        config += ' --angularEnabled=1 '
    else:
        config += ' --angularEnabled=0 '

    # headers
    headers = ['Total bitstream size', 'positions bitstream size', 
                'colors bitstream size', 'reflectances bitstream size',
                'Processing time (user)', 'Processing time (wall)',
                'positions processing time (user)', 
                'colors processing time (user)', 
                'reflectances processing time (user)']

    subp=subprocess.Popen(rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=0' + config + \
                        ' --uncompressedDataPath='+filedir + \
                        ' --compressedStreamPath='+bin_dir, 
                        shell=True, stdout=subprocess.PIPE)

    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()
    
    return results

def gpcc_decode(bin_dir, rec_dir, cfgdir=None, attr=True, test_geo=True, test_attr=True, show=False, version=19):
    if attr: config = ' --convertPlyColourspace=1'
    else: config = ''
    if cfgdir is not None:
        config = ' --config='+cfgdir

    
    subp=subprocess.Popen(rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=1'+ config + \
                            ' --compressedStreamPath='+bin_dir + \
                            ' --reconstructedDataPath='+rec_dir + \
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    
    headers = ['Total bitstream size', 'positions bitstream size', 
                'colors bitstream size', 'reflectances bitstream size', 
                'Processing time (user)', 'Processing time (wall)',
                'positions processing time (user)', 
                'colors processing time (user)', 
                'reflectances processing time (user)']
    
    results = {}
    c=subp.stdout.readline()
    while c:
        if show: print(c)   
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()
    
    return results