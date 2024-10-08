
import os, sys
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
from data_utils.geometry.inout import read_ply_o3d, write_ply_o3d
import subprocess
import time
import numpy as np
import open3d as o3d
rootdir_tmc13 = os.path.split(__file__)[0]


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def pc_error(infile1, infile2, resolution, normal=False, show=False, details=False):
    # headersF = ["mse1      (p2point)", "mse1,PSNR (p2point)", 
    #            "h.       1(p2point)", "h.,PSNR  1(p2point)",
    #            "mse2      (p2point)", "mse2,PSNR (p2point)", 
    #            "h.       2(p2point)", "h.,PSNR  2(p2point)" ,
    #            "mseF      (p2point)", "mseF,PSNR (p2point)", 
    #            "h.        (p2point)", "h.,PSNR   (p2point)" ]
    # headersF_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
    #                   "mse2      (p2plane)", "mse2,PSNR (p2plane)",
    #                   "mseF      (p2plane)", "mseF,PSNR (p2plane)"]             
    headers = ["mseF      (p2point)", "mseF,PSNR (p2point)"]
    if details: headers += ["mse1      (p2point)", "mse1,PSNR (p2point)",
                            "mse2      (p2point)", "mse2,PSNR (p2point)"]

    command = str(rootdir_tmc13+'/pc_error_d' + 
                          ' -a '+infile1+ 
                          ' -b '+infile2+ 
                          ' --hausdorff=1 '+ 
                          ' --resolution='+str(resolution))
    if normal:
        headers +=["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
        command = str(command + ' -n ' + infile1)
    results = {}   
    subp=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')# python3.
        if show: print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c=subp.stdout.readline() 


    return results

def chamfer_dist(a, b):
    pcdA = o3d.geometry.PointCloud()
    pcdA.points = o3d.utility.Vector3dVector(a.astype('float32'))
    pcdB = o3d.geometry.PointCloud()
    pcdB.points = o3d.utility.Vector3dVector(b.astype('float32'))
    distA = pcdA.compute_point_cloud_distance(pcdB)
    distB = pcdB.compute_point_cloud_distance(pcdA)
    distA = np.array(distA)**2
    distB = np.array(distB)**2

    return distA, distB


    

