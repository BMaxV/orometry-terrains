import argparse
import os
import random
import numpy as np
from scipy.spatial import cKDTree

def main():
    args = arg_parse()
    fiso,fpro = old_open(args.isolationList,args.prominenceList)
    
    isos, vals_overall = read_old_files(fiso,fpro)
    
    my_outputlines = process(isos,vals_overall)
    
    write_output(args,my_outputlines)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("isolationList", help="isolation file")
    parser.add_argument("prominenceList", help="prominence file")
    parser.add_argument("outFile", help="output file")
    parser.add_argument("--fileHeaders", help="files contain headers", action='store_true')
    parser.add_argument("--deleteOriginals", help="delete isolation and prominence list files", action='store_true')
    args = parser.parse_args()
    return args
    
def old_open():
    fiso = open(args.isolationList)
    if args.fileHeaders:
        fiso.readline()
    fpro = open(args.prominenceList)
    if args.fileHeaders:
        fpro.readline()
    return fiso,fpro

def make_output(args,output_data):
    with open(args.outFile, 'w') as f:
        fout.write('latitude,longitude,elevation in feet,key saddle latitude,key saddle longitude,prominence in feet,isolation latitude,isolation longitude,isolation in km\n')
        real_output_string = "\n".join(output_data)
        f.write(real_output_string)


def read_old_files(fiso,fpro):
    print('Reading isolations')
    isos = []
    for line in fiso:
        isos.append([float(x) for x in line.split(',')])
        
    vals_overall = []
    for line in fpro:
        vals = [float(x) for x in line.split(',')]
        vals_overall.append(vals)
    
    return isos, vals_overall

def process(isos, vals_overall):

    # I actually don't care about the isos
    ni = len(isos)
    
    isoPoints = np.zeros((ni, 2))
    for i in range(ni):
        isoPoints[i,0] = isos[i][0]
        isoPoints[i,1] = isos[i][1]
    # I care about the KD tree. to sample it...
    kdIsos = cKDTree(isoPoints)
    
    output_lines = ['latitude,longitude,elevation in feet,key saddle latitude,key saddle longitude,prominence in feet,isolation latitude,isolation longitude,isolation in km\n']
    output_data = []
    
    print('Merging isolation and prominence lists')
    numPeaks = 0
    numIsolated = 0
    
    # vals_overall are the p_100 values? the naming...
    for vals in vals_overall:
        
        minDist = (3/3600)*3.3 # about 300m
        match = None
        
        # ...here.
        nndist, nnid = kdIsos.query(np.array([vals[0], vals[1]]), k=1)
        
        dist_match_1 = nndist < 0.25*minDist
        dist_match_2 = nndist < minDist
        dist_match = dist_match_1 or dist_match_2
        
        other_match = abs(vals[2] - isos[nnid][2]) < 200
        
        if dist_match and other_match:
            match = isos[nnid]
            
        if match:
            #fout.write('%s,%.4f,%.4f,%.4f\n' % (line.strip(), match[3], match[4], match[5]))
            
            # ", ".join(vals) restringifies. which is probably also a bad idea. WE DONT WANT THE STRINGS
            data_tup = (match[3], match[4], match[5])
            strvals = [str(x) for x in vals] # I don't get why we are doing the string thing here, what do we need it for?
            output_lines.append('%s,%.4f,%.4f,%.4f\n' % (", ".join(strvals), *data_tup ))
            output_data.append(data_tup)
        else:
            #fout.write('%s,%.4f,%.4f,%.4f\n' % (line.strip(), vals[0] + 3/3600*(random.random()*2-1), vals[1] + 3/3600*(random.random()*2-1), 0.1))
            
            # ALSO, WHY IS THIS RANDOMIZING THE INPUTS?!
            data_tup = (vals[0] + 3/3600*(random.random()*2-1), vals[1] + 3/3600*(random.random()*2-1), 0.1)
            
            strvals = [str(x) for x in vals] # I don't get why we are doing the string thing here, what do we need it for?
            output_lines.append('%s,%.4f,%.4f,%.4f\n' % (", ".join(strvals), *data_tup))
            output_data.append(data_tup)
            numIsolated += 1
        
        numPeaks += 1
    
    # numPeaks and numIsolated are not used here, idk what the point
    # was
    # there was a print... for like, semi manual confirmation or something?
    # come on...
    
    return output_lines, output_data
    

if __name__=="__main__":
    main()
