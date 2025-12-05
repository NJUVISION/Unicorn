# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import subprocess
import time
import os, sys
rootdir_tmc13 = os.path.split(__file__)[0]
rootdir_cfg = os.path.join(os.path.split(__file__)[0], 'gpcc_cfg')


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number


def gpcc_encode(filedir, bin_dir, posQuantscale=1, version=14, cfgdir='dense.cfg', DBG=False):
    cfgdir = os.path.join(rootdir_cfg, cfgdir)
    assert os.path.exists(cfgdir)
    # print('DBG!!! GPCC', version, cfgdir)

    cmd = rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=0 ' \
        + ' --config='+cfgdir \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # subp.wait()
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if DBG: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline()

    return results

def gpcc_decode(bin_dir, dec_dir, version=14, DBG=False):
    cmd = rootdir_tmc13+'/tmc3_v'+str(version)+' --mode=1 ' \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # subp.wait()
    headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    results = {}
    c=subp.stdout.readline()
    while c:
        if DBG: print(c)   
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1: 
                value = number_in_line(line)
                results[key] = value   
        c=subp.stdout.readline()

    return results
