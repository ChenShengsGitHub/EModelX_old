import numpy as np
from skimage import transform
import numpy as np
import numpy as np
import mrcfile
import os
import pdb

from static_config import static_config

chainID_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

AA_types = {"ALA":1,"CYS":2,"ASP":3,"GLU":4,"PHE":5,"GLY":6,"HIS":7,"ILE":8,"LYS":9,"LEU":10,"MET":11,"ASN":12,"PRO":13,"GLN":14,"ARG":15,"SER":16,"THR":17,"VAL":18,"TRP":19,"TYR":20}
AA_T = {AA_types[k]-1: k for k in AA_types}
AA_abb_T = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y"}
AA_abb = {AA_abb_T[k]:k for k in AA_abb_T}
abb2AA = {"A":"ALA","C":'CYS',"D":'ASP',"E":'GLU',"F":'PHE',"G":'GLY',"H":"HIS","I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"}

def get_info_from_csv(csv_data):
    emid = str(csv_data[0])
    date = str(csv_data[1])
    resol = float(csv_data[2])
    pdbid = str(csv_data[3])
    while len(emid) < 4:
        emid = '0' + emid
    return emid,date,resol,pdbid

def visualize(map,path,offset):
    with mrcfile.new(path,data=map.astype(np.float32), overwrite=True) as image_mrc:
        image_mrc.header.nzstart = offset[0]
        image_mrc.header.nystart = offset[1]
        image_mrc.header.nxstart = offset[2]
        image_mrc.header.maps = 1
        image_mrc.header.mapr = 2
        image_mrc.header.mapc = 3
    

def transpose(numpy_image, axis_order, offset):
    trans_offset = []
    trans_order = []
    for i in range(3):
        for j in range(len(axis_order)):
            if axis_order[j] == i:
                trans_offset.append(offset[j])
                trans_order.append(j)
    image = np.transpose(numpy_image, trans_order)

    return image, trans_offset


def reshape(numpy_image, offset, pixel_size):
    if pixel_size == [1,1,1]:
        return numpy_image, offset
    image = transform.rescale(numpy_image, pixel_size)
    for i in range(len(offset)):
        offset[i] *= pixel_size[i]
    return image, offset


def normalize(numpy_image,offset):
    np.nan_to_num(numpy_image)
    median = np.median(numpy_image)
    image = (numpy_image > median) * (numpy_image - median)
    
    vlid_coords = np.array(np.where(image>0))
    minX = np.min(vlid_coords[0])
    maxX = np.max(vlid_coords[0])
    minY = np.min(vlid_coords[1])
    maxY = np.max(vlid_coords[1])
    minZ = np.min(vlid_coords[2])
    maxZ = np.max(vlid_coords[2])
    image = image[minX:maxX+1,minY:maxY+1,minZ:maxZ+1]
    # print('origin shape',image.shape,'new shape', image.shape)
    minXYZ = [minX,minY,minZ]
    offset = [offset[0]+minXYZ[0], offset[1]+minXYZ[1], offset[2]+minXYZ[2]]
    
    p999 = np.percentile(image[np.where(image > 0)], 99.9)
    if p999 != 0:
        image = (image < p999) * image + (image >= p999) * p999
        image /= p999
        return image, offset
    else:
        print('normalization error!!!')
        return


def processEMData(EMmap):
    em_data = np.array(EMmap.data)
    pixel_size = [float(EMmap.header.cella.x / EMmap.header.mx),
                  float(EMmap.header.cella.y / EMmap.header.my),
                  float(EMmap.header.cella.z / EMmap.header.mz)]
    axis_order = [int(EMmap.header.maps) - 1, int(EMmap.header.mapr) - 1,
                  int(EMmap.header.mapc) - 1]
    offset = [float(EMmap.header.nzstart), float(EMmap.header.nystart),
              float(EMmap.header.nxstart)]
    # print(pixel_size,axis_order,offset,end='\t')
    em_data, offset = transpose(em_data, axis_order, offset)
    em_data, offset = reshape(em_data, offset, pixel_size)
    em_data, offset = normalize(em_data, offset)
    return em_data, offset


def calc_dis(distanceList1,distanceList2):
    y = [distanceList2 for _ in distanceList1]
    y = np.array(y)
    x = [distanceList1 for _ in distanceList2]
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a

def parseMMscore(gt_pdb,pred_pdb):
    lines = os.popen(f'{static_config.MMalign} {gt_pdb} {pred_pdb}').readlines()
    ResNum_pdb,ResNum_pred,MM1,MM2,RMSD,SeqID=9999,9999,9999,9999,9999,9999
    for line in lines:
        if len(line)>len('Length of Structure_1') and line[:len('Length of Structure_1')]=='Length of Structure_1':
            ResNum_pdb=int(line.split(':')[1].split('residues')[0])
        elif len(line)>len('Length of Structure_2') and line[:len('Length of Structure_2')]=='Length of Structure_2':
            ResNum_pred=int(line.split(':')[1].split('residues')[0])
        elif len(line)>len('Aligned length=') and line[:len('Aligned length=')]=='Aligned length=':
            RMSD=float(line.split('RMSD=')[1].split(',')[0])
            SeqID=float(line.split('n_aligned=')[1].strip())
        elif line.find('normalized by length of Structure_1')!=-1:
            MM1=float(line.split('TM-score=')[1].split('(')[0])
        elif line.find('normalized by length of Structure_2')!=-1:
            MM2=float(line.split('TM-score=')[1].split('(')[0])
    return ResNum_pdb,ResNum_pred,MM1,MM2,RMSD,SeqID
