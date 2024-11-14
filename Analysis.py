import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from utils import coords
from utils import shapefiles
from analysis import peaksdata
from analysis import filterPoints
from analysis import mergePeakLists


def data_check():
    # Download prominence and isolation lists from Andrew Kirmse project:
    # https://github.com/akirmse/mountains

    # path to prominence and isolation files
    prominenceDB = 'data/prominence-p100.txt'
    isolationDB  = 'data/alliso-sorted.txt'

    if not os.path.exists(prominenceDB) or not os.path.exists(isolationDB):
        print('ERROR: peak databases not found!')


    # region shapefiles
    regionShapesDir = 'data/regionShapes'
    regionShapes = [f for f in os.listdir(regionShapesDir) if f.endswith('.shp')]


    # # Filter and unify prominence and isolation peak lists
    regionPeaksDir = 'data/regionPeaks'

    if not os.path.exists(regionPeaksDir):
        os.makedirs(regionPeaksDir)
    
    
    regionStatsDir = 'data/regionStats'

    if not os.path.exists(regionStatsDir):
        os.makedirs(regionStatsDir)
    
    return regionShapes,regionShapesDir,regionPeaksDir,regionStatsDir


def my_filter(regionShapes,regionShapesDir,regionPeaksDir,output_write = False):
    """
    process each region to filter the database peaks that are inside
    since this process takes a long time, we provide the functions as standalone scripts for batch processing
    """
    output_data_d = {}
    for region in regionShapes:
        print(region)
        
        print(' 1/3 Filtering prominence DB...')
        shapes_filename = os.path.join(regionShapesDir, region)
        
        points_filename1 = "data/prominence-p100.txt"
        points_filename2 = "data/alliso-sorted.txt"
        
        # I don't want strings, really... do I?
        
        output_string_prom, output_data_prom = filterPoints.core_without_output(shapes_filename,points_filename1)
        #subprocess.call('python3 ./analysis/filterPoints.py "%s" data/prominence-p100.txt prom.txt' 
        #                % (os.path.join(regionShapesDir, region)))
        
        print(' 2/3 Filtering isolation DB...')
        output_string_iso, output_data_iso = filterPoints.core_without_output(shapes_filename,points_filename2)
        #subprocess.call('python3 ./analysis/filterPoints.py "%s" data/alliso-sorted.txt isol.txt' 
        #                % (os.path.join(regionShapesDir, region)))

        # merge lists
        print(' 3/3 Merging lists...')
        output_lines, output_data = mergePeakLists.process(output_data_iso,output_data_prom)
        #subprocess.call('python3 ./analysis/mergePeaklists.py isol.txt prom.txt tmppeaks.csv --deleteOriginals')
        
        output_data_d[region] = output_data
        output_write = True
        # move results to output dir
        if output_write:
            path = os.path.join(regionPeaksDir, region.replace(".shp", ".csv"))
            with open(path,"w") as f:
                f.write("".join(output_lines)) # new lines are built in.
        #shutil.move("tmppeaks.csv", os.path.join(regionPeaksDir, region.replace(".shp", ".csv")))
        
        print("breaking for test reasons")
        break # for test reasons
        
    return output_data

def get_header_line(distributions,writeFeatures):
    """
    changes from original, replaced fout.write with string concatonation
    future todo: f-strings + naming
    """

    #def writeHeaderToFile(distributions):    
    fout_line_string = 'lat,lon,peaks'
    #fout.write('lat,lon,peaks')
    for feat in writeFeatures:
        if feat in ['elevation', 'prominence', 'isolDir', 'saddleDir']:
            for val in distributions[feat]['bins'][:-1]:
                #fout.write(',%s_%d' % (feat, int(val)))
                fout_line_string += ',%s_%d' % (feat, int(val))
        elif feat == 'domGroup':
            for val in distributions[feat]['bins'][:-1]:
                #fout.write(',%s_%.2f' % (feat, 100*val))
                fout_line_string += ',%s_%.2f' % (feat, 100*val)
        else:
            for val in distributions[feat]['bins'][:-1]:
                #fout.write(',%s_%.2f' % (feat, val))
                fout_line_string += ',%s_%.2f' % (feat, val)
            
    #fout.write('\n')
    fout_line_string += "\n"
    return fout_line_string

def make_location_stats_line(lat,lon,npeaks,distributions,writeFeatures):
    """changes from original:
    replaced file io with string concat."""
    
    #def writeLocationStatsToFile(fout, lat, lon, npeaks, distributions):
    
    line = '%.4f,%.4f,%d'%(lat, lon, npeaks)
    #fout.write('%.4f,%.4f,%d'%(lat, lon, npeaks))
    for feat in writeFeatures:
        for val in distributions[feat]['hist']:
            line+=',%d' % val
            #fout.write(',%d' % val)
    #fout.write('\n')
    line += "\n"
    #print("stats line", line)
    return line

def process_regionShapes(regionShapes,regionShapesDir,regionPeaksDir,regionStatsDir,diskRadius,writeFeatures):
    """
    changes from original:
    changed initialization for the output files to outside of the loop
    changed file opening and writing to one with open; write each.
    
    increased isolation of file io
    """
    
    distribution_outputs = []
    
    # process each region (note: it takes a long time!)
    for region in regionShapes:
        
        # sample stats locations inside polygon, separated at least 1/2 radius distance
        sampleLocations = shapefiles.sampleShapefileLocations(os.path.join(regionShapesDir, region), diskRadius)

        # region peaks DB
        df = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
        df = peaksdata.addExtraColumns(df) # what is this, why do this?
        
        
        # normalize distance columns
        df['isolation']  /= diskRadius
        df['saddleDist'] /= diskRadius
        
        # results file
        output_fn = os.path.join(regionStatsDir, region.replace('.shp', '.csv'))
        
        distribution_output_string = ""
        # compute statistics
                
        for di,diskCenter in enumerate(sampleLocations):
            # filter peaks in disk using haversine distance
            peaks = peaksdata.filterPeaksHaversineDist(df, diskCenter, diskRadius)
            
            # skip if not enough peaks
            if peaks.shape[0] < 20:
                continue
            
            # compute statistics
            # diskRadius = 1   to have isolation/saddle dist histograms axis from 0 to 1, note we normalized distances before
            # detailed = False for the classification histograms, for synthesis we double the number of bins
            distributions = peaksdata.computeDistributions(peaks, diskRadius=1.0, detailed=False)
            
            if len(distribution_output_string)==0:
                #writeHeaderToFile(fout, distributions)
                header_line = get_header_line(distributions,writeFeatures)
                distribution_output_string += header_line
            
            # write data line
            #writeLocationStatsToFile(fout, diskCenter[0], diskCenter[1], peaks.shape[0], distributions)
            line = make_location_stats_line(diskCenter[0], diskCenter[1], peaks.shape[0], distributions,writeFeatures)
            distribution_output_string += line 
            print('%s: %3d/%3d samples'%(region, di+1, len(sampleLocations)), end='\r' if di+1 < len(sampleLocations) else '\n')
        
        distribution_outputs.append((output_fn,distribution_output_string))
        
        print("breaking for test purposes")
        break
    print('calculation done!')
    
    write_distribution_outputs(distribution_outputs)
    
    print('output writing done!')
    
def write_distribution_outputs(distribution_outputs):
    for my_tuple in distribution_outputs:
        output_fn, out_string = my_tuple
        with open(output_fn,"w") as f:
            f.write(out_string)

def make_csv_result(regionStatsDir,diskRadius,normalize = True):
    """
    if normalize is True: write frequencies, otherwise keep histogram counts
    we observed that frequencies work better in the classifier
    """
        
        # file where the dataset will be stored
    fileDataset = 'data/regions_%dkm.csv' % (int(diskRadius))

    # regions to put in the dataset (for example, we could omit certain regions, we can also do it later in classifier)
    datasetRegions = [f for f in os.listdir(regionStatsDir) if f.endswith('.csv')]

    
    alldf = []
    for file in datasetRegions:
        
        # name
        terrainName = file.split('.')[0]
        
        # read dataframe
        file_path = os.path.join(regionStatsDir, file)
        print(file_path)
        df = pd.read_csv(file_path)
        
        # keep number of peaks
        npeaks = df['peaks'].values
        
        # drop lat, lon, npeaks
        df.drop(['lat', 'lon', 'peaks'], axis=1, inplace=True)
        
        # normalize histogram columns?
        if normalize:
            for c in df.columns:
                df[c] = df[c].astype(np.float32)/npeaks
        
        # add terrain name column
        df.insert(0, 'terrain', terrainName)
        alldf.append(df)
        
        print('%4d %s' % (df.shape[0], terrainName))
        
    alldf = pd.concat(alldf, ignore_index=True)
    alldf.to_csv(fileDataset, float_format='%.4f', index=False)

def main():
    
    regionShapes,regionShapesDir,regionPeaksDir,regionStatsDir = data_check()
    #peak_dict = my_filter(regionShapes,regionShapesDir,regionPeaksDir)
    # # Compute statistics 
    

    # statistics disk radius
    diskRadius = 30

    writeFeatures = ['elevation', 'elevRel', 'prominence', 'promRel', 
                     'dominance', 'domGroup', 'relevance',
                     'isolation', 'isolDir', 'saddleDist', 'saddleDir']
    
    process_regionShapes(regionShapes,regionShapesDir,regionPeaksDir,regionStatsDir,diskRadius,writeFeatures)
    
    make_csv_result(regionStatsDir,diskRadius)

if __name__=="__main__":
    main()



