import numpy as np 
import os
import subprocess
rootdir_extension = os.path.split(__file__)[0]

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item) 
        except ValueError:
            continue
        
    return number

# =/home/temp/wjq/backup_new/common/extension/PCQM/build/PCQM

def get_pcqm(infile1, infile2, show=True):
    """infile1: ground truth
        infile2: lossy
    """
    command = rootdir_extension + '/PCQM' + \
    ' '+infile1+ \
    ' '+infile2 + \
    ' -r 0.004 -knn 20 -rx 2.0 '

    headers = "PCQM value is :"
   
    subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    # subp.communicate(input="\n")

    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show:
            print(line)
        if line.find(headers) != -1:
            value = number_in_line(line)
        c=subp.stdout.readline() 

    return value