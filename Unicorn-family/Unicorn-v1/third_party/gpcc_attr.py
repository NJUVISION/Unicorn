# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2023-01-07

import os, sys, time
import numpy as np
import subprocess
import glob
import os
import subprocess

import numpy as np
import open3d as o3d
import pandas as pd
# from pypcd import pypcd
from tqdm import tqdm
import torch
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
        try:
            number = float(item)
        except ValueError:
            continue

    return number


########################################################## v14,v19,20,21 ###########################################################
def gpcc_encode(filedir, bin_dir, posQuantscale=1, transformType=0, qp=22,
                cfgdir=None, test_time=False, show=True, version=21, attribute='color', angularEnabled=0):
    """
    """
    config = ' --trisoupNodeSizeLog2=0' + \
             ' --neighbourAvailBoundaryLog2=8' + \
             ' --intra_pred_max_node_size_log2=6' + \
             ' --maxNumQtBtBeforeOt=4' + \
             ' --planarEnabled=1' + \
             ' --planarModeIdcmUse=0' + \
             ' --minQtbtSizeLog2=0' + \
             ' --positionQuantizationScale=' + str(posQuantscale)
    # lossless
    if posQuantscale == 1:
        config += ' --mergeDuplicatedPoints=0' + \
                  ' --inferredDirectCodingMode=1'
    else:
        config += ' --mergeDuplicatedPoints=1'
    if qp is not None:
        if version >= 14:
            if transformType == 0:
                # RAHT
                config += ' --convertPlyColourspace=1' + \
                          ' --transformType=0' + \
                          ' --qp=' + str(qp) + \
                          ' --qpChromaOffset=0' + \
                          ' --bitdepth=8' + \
                          ' --attrOffset=0' + \
                          ' --attrScale=1' + \
                          ' --attribute=' + attribute
            if transformType == 1:
                # predict (for lossless coding)
                config += ' --convertPlyColourspace=1' + \
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
                          ' --attribute=' + attribute

        else:
            if transformType == 0:
                # RAHT
                if version == 13:
                    config += ' --convertPlyColourspace=1' + \
                              ' --transformType=1' + \
                              ' --qp=' + str(qp) + \
                              ' --qpChromaOffset=1' + \
                              ' --bitdepth=8' + \
                              ' --attribute=' + attribute
                if version == 12:
                    config += ' --convertPlyColourspace=1' + \
                              ' --transformType=0' + \
                              ' --qp=' + str(qp) + \
                              ' --qpChromaOffset=0' + \
                              ' --bitdepth=8' + \
                              ' --attrOffset=0' + \
                              ' --attrScale=1' + \
                              ' --attribute=' + attribute

        if transformType == 2:
            # predict & lifting
            config += ' --convertPlyColourspace=1' + \
                      ' --transformType=2' + \
                      ' --numberOfNearestNeighborsInPrediction=3' + \
                      ' --levelOfDetailCount=12' + \
                      ' --lodDecimator=0' + \
                      ' --adaptivePredictionThreshold=64' + \
                      ' --qp=' + str(qp) + \
                      ' --qpChromaOffset=0' + \
                      ' --bitdepth=8' + \
                      ' --attrOffset=0' + \
                      ' --attrScale=1' + \
                      ' --attribute=' + attribute

        # # for goemetry  planarModeIdcmUse=32!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # config+=' --planarModeIdcmUse=32' + \
        #         ' --partitionMethod=0' +\
        #         ' --maxNumQtBtBeforeOt=6'+ \
        #         ' --spherical_coord_flag=1'

    if cfgdir is not None:
        config = ' --config=' + cfgdir

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

    # headers = ['positions bitstream size', 'Total bitstream size']
    # if test_time:
    #     headers += ['positions processing time (user)', 'Processing time (user)', 'Processing time (wall)']
    #     headers += ['colors processing time (user)', 'reflectances processing time (user)']
    # if qp is not None: headers += ['colors bitstream size']
    # if qp is not None and test_time:  headers += ['colors processing time (user)']

    #
    # print('DBG!!!config'*10, config)
    subp = subprocess.Popen('./tmc3_v' + str(version) + ' --mode=0' + config + \
                            ' --uncompressedDataPath=' + filedir + \
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)

    results = {}
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    return results


def gpcc_decode(bin_dir, rec_dir, cfgdir=None, attr=True, test_geo=True, test_attr=True, show=False, version=19):
    if attr:
        config = ' --convertPlyColourspace=1'
    else:
        config = ''
    if cfgdir is not None:
        config = ' --config=' + cfgdir

    subp = subprocess.Popen('./tmc3_v' + str(version) + ' --mode=1' + config + \
                            ' --compressedStreamPath=' + bin_dir + \
                            ' --reconstructedDataPath=' + rec_dir + \
                            ' --outputBinaryPly=0',
                            shell=True, stdout=subprocess.PIPE)
    # headers
    # headers = []
    # if test_geo: headers += ['positions bitstream size', 'positions processing time (user)',
    #             'Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    # if test_attr: headers += ['colors bitstream size', 'colors processing time (user)']
    headers = ['Total bitstream size', 'positions bitstream size',
               'colors bitstream size', 'reflectances bitstream size',
               'Processing time (user)', 'Processing time (wall)',
               'positions processing time (user)',
               'colors processing time (user)',
               'reflectances processing time (user)']

    results = {}
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        line = c.decode(encoding='utf-8')
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value
        c = subp.stdout.readline()

    return results


def read_ply_ascii_geo(filedir):
    files = open(filedir)

    data = []
    for i, line in enumerate(files):
        wordslist = line.split(" ")
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == "\n":
                    continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)

    data = np.array(data)

    coords = data[:, 0:3].astype("float32")
    refl = data[:, 3:4].astype("int16")

    return coords, refl


def pc_error(infile1, infile2, res, normal=True, color=False, lidar=True, show=False):
    # Symmetric Metrics. D1 mse, D1 hausdorff.
    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)", "h.        (p2point)", "h.,PSNR   (p2point)"]
    haders_p2plane = ["mseF      (p2plane)", "mseF,PSNR (p2plane)"]
    headers = headersF

    command = "./pc_error_d" + " -a " + infile1 + " -b " + infile2 + " --hausdorff=1 " + " --resolution=" + str(res)
    if normal:
        headers += haders_p2plane
        command += " -n " + infile1
    if color:
        command += " --color=1"
    if lidar:
        command += " --lidar=1"
    headersF_color = ["  c[0],PSNRF", "  c[1],PSNRF", "  c[2],PSNRF", "   r,PSNR   F"]
    headersF_color_mse = ["  c[0],    F", "  c[1],    F", "  c[2],    F"]
    headers += headersF_color
    headers += headersF_color_mse

    results = {}

    subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding="utf-8")  # python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c = subp.stdout.readline()

    return results


def write_ply_ascii(filedir, coords, feats, dtype_coords='int32', dtype_feats='int32'):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    if feats.shape[-1]==3:
        f.writelines(['property uchar red\n','property uchar green\n','property uchar blue\n'])
    if feats.shape[-1]==1:
        f.writelines(['property uint16 reflectance\n'])
    f.write('end_header\n')
    coords = coords.astype(dtype_coords)
    if feats.shape[-1]==3:
        feats = np.clip(feats, 0, 255).astype(dtype_feats)
        for xyz, rgb in zip(coords, feats):
            f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                        str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    if feats.shape[-1]==1:
        feats = feats.astype(dtype_feats)
        for xyz, r in zip(coords, feats):
            f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                        str(r[0]), '\n'])
    f.close()

    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src_dir", default="/media/ivc3090ti/disk1/zjz/data/kitti_test/11")
    parser.add_argument("--tgt_dir", default="./results/qp20")
    parser.add_argument("--gqs", type=float, default=1, help="geom quant step")
    parser.add_argument("--rqs", type=int, default=20, help="relf quant step")
    parser.add_argument("--res", type=float, default=59.70, help="resolution for pc_error")
    args = parser.parse_args()

    src_dir = glob.glob(f"{args.src_dir}/**/*.ply", recursive=True)
    src_dir = sorted(src_dir)
    src_dir = src_dir[:10]
    # src_dir = ['1698394383499.pcd']



    ori_dir = f"{args.tgt_dir}/ori"
    bin_dir = f"{args.tgt_dir}/bin"
    qut_dir = f"{args.tgt_dir}/qut"
    dec_dir = f"{args.tgt_dir}/dec"

    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(qut_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)

    csv_filename = os.path.join(args.tgt_dir, "results.csv")

    for i, file_dir in enumerate(tqdm(src_dir)):
        file_name = file_dir.split("/")[-1].split(".")[0]
        coords, reflc = read_ply_ascii_geo(file_dir)
        reflc= np.zeros_like(reflc)
        points = coords.shape[0]
        results = {"file_name": file_name}

        ori_file = f"{ori_dir}/{file_name}_ori.ply"
        bin_file = f"{bin_dir}/{file_name}_enc.bin"
        qut_file = f"{bin_dir}/{file_name}_qut.ply"
        dec_file = f"{dec_dir}/{file_name}_dec.ply"
        write_ply_ascii(qut_file, coords, reflc)
        # quant to 1mm and convert to ply
        # coords, refl = read_pcd(file_dir)
        # write_ply_o3d_normal(ori_file, coords, refl)
        # coords = np.round(coords / 0.001)
        # coords, idx = np.unique(coords, axis=0, return_index=True)
        # refl = refl[idx]
        # write_ply_o3d_normal(qut_file, coords, refl)

        # encode
        enc_results = gpcc_encode(file_dir, bin_file, posQuantscale=args.gqs, qp=args.rqs, transformType=0, version=21,
                                  attribute='reflectance')

        # decode
        dec_results = gpcc_decode(bin_file, dec_file, version=21)

        # dequant to origin
        # coords, refl = read_ply_ascii_geo(dec_file)
        # coords = coords * 0.001
        # refl = refl.squeeze(-1)
        # write_ply_o3d(dec_file, coords, refl)

        # pc_error
        pc_error_results = pc_error(file_dir, dec_file, res=args.res)

        # collect results
        results["geom qs"] = args.gqs
        results["refl qs"] = args.rqs

        results["bpp geom"] = enc_results["positions bitstream size"] * 8 / points
        results["bpp refl"] = enc_results["reflectances bitstream size"] * 8 / points
        results["bpp sum"] = enc_results["Total bitstream size"] * 8 / points

        results["PSNR D1"] = pc_error_results["mseF,PSNR (p2point)"]
        results["PSNR D2"] = pc_error_results["mseF,PSNR (p2plane)"]
        results["PSNR refl"] = pc_error_results["   r,PSNR   F"]

        results = pd.DataFrame([results])

        if i == 0:
            results_allfile = results.copy(deep=True)
        else:
            results_allfile = pd.concat([results_allfile, results], ignore_index=True)

        results_allfile.to_csv(csv_filename, index=False)
