import os
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from shapely.geometry import Polygon

from utils.poisson import PoissonDisc
from utils import coords
from utils import erosion_inputs
from utils.divtree_reader import readDivideTree
from utils.meshwriter import writeTerrainMesh
from synthesis.divtree_to_dem import *

from scipy.spatial import Delaunay


def mesh_main():
    np.random.seed(42)
    
    
    terrainName = "synthesis_andes_peru"
    # this is the name that's used to look up the input files.
    
    """
    longer comment
    
    in *principle* it should be possible to just drive
    everything directly without resorting to file IO
    file IO is only done to allow more piece wise
    generation / analysis.
    it's just a dislike of mine I guess...
    
    file IO is a crutch. It's additional functions, built because
    either our handling of the data or the hardware is not
    good enough. Which is fine, as a "limitation handling method",
    but it should be possible to run the code in a "pure" way
    that only needs real earth data, and the target size as inputs
    and produces my mesh or height profile or whatever.
    """
    
    # reading inputs
    coordinates_elevation, terrainSize, poissonSamples, parameters = mesh_setup(terrainName)
    
    # unpacking
    [peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree] = coordinates_elevation 
    [reconsParams,sourcesParams] = parameters
    
    # ### Mesh
    meshVerts, meshElevs, meshTris, debugInfo = divideTreeToMesh(peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks,
                                                                 terrainSize, poissonSamples, {**reconsParams, **sourcesParams})
    print('mesh calculation done!')

    # Geometry as mesh, exact representation of the terrain we synthesized,
    # it contains all peaks and saddles and the elevation is exact

    xMax = terrainSize[0]
    yMax = terrainSize[1]
    bbox = Polygon([[0, 0], [0, yMax], [xMax, yMax], [xMax, 0]])

    #writeTerrainMesh(terrainName + '_mesh.obj', meshVerts.copy(), np.maximum(0, meshElevs), meshTris.copy(), bbox)

    print(f"mesh writing done, mesh elevation range {meshElevs.max()}, {meshElevs.min()}")
    
    # debug and other visualizations
    image_outputs(meshVerts, meshElevs, terrainSize, terrainName, coordinates_elevation, debugInfo)

def image_outputs(meshVerts, meshElevs, terrainSize,terrainName, coordinates_elevation, debugInfo):
    
    [peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree] = coordinates_elevation 
    
    if False:
        hfsize = erosion_inputs.image_settings(terrainSize)
        hf = erosion_inputs.heightfield_delaunay(meshVerts, meshElevs, terrainSize, hfsize)
        erosion_inputs.make_delauney_picture(hf, terrainName)
        erosion_inputs.distance_fields(hfsize, terrainName,terrainSize,peakCoords,saddleCoords,meshVerts,debugInfo)
        print('heightfield and delauney done')
    
    # the other "debug" visualiziations
    
    xmin = ymin = 0
    xmax = terrainSize[0]
    ymax = terrainSize[1]
    
    dimensions = [xmin,ymin,xmax,ymax]
    
    voronoi_and_river_lines(dimensions,saddlePeaks,peakCoords,saddleCoords,debugInfo)
    print('voronoi rivers, done')
    coarse_river_network(dimensions,saddlePeaks,peakCoords,saddleCoords,debugInfo)
    print('coarse river network done')
    refined_networks(dimensions,meshVerts,debugInfo)
    print('refined networks done')
    poisson_something(dimensions,meshVerts,debugInfo)
    print('poisson something done')

def mesh_setup(terrainName):
    # read divide tree
    
    peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree = readDivideTree(terrainName + '_dividetree.txt')
    peakCoords   = coords.deg2km(peakCoords)
    saddleCoords = coords.deg2km(saddleCoords)
    
    # Terrain size, we could also compute an extended bbox of the peak/saddle positions in the divide tree
    terrainSize = [100, 100]

    out_of_bounds_check(peakCoords,saddleCoords,terrainSize)

    # Minimum elevation value we want this terrain to have
    minTerrainElev = 0.5*saddleElevs.min()

    helper_print_out = [f'Size in km {terrainSize}',
    f'#Peaks {peakElevs.size}',
    f'#Saddles {saddleElevs.size}',
    f'Peaks  elevation range {peakElevs.min()}, {peakElevs.max()}',
    f'Saddle elevation range {saddleElevs.min()}, {saddleElevs.max()}',
    f'Base terrain elevation {minTerrainElev}',
    ]
    
    print("\n".join(helper_print_out))
    
    poissonSamples = generate_poisson_samples(terrainSize)

    # reconstruction algorithm parameters (in practice, we only tune maxSlopeCoeff, if any)
    reconsParams = {
        'minTerrainElev': minTerrainElev, # clip elevations lower than this
        'maxSlopeCoeff': 0.5,             # linear factor for the river slope, lower = gentler slopes

        'refineDistance': 0.12,           # resampling distance for fine river/ridge networks (affects performance)
        'riversPerturbation': 0.20,       # planar perturbation of new river nodes, as % of edge length
        'ridgesPerturbation': 0.15,       # planar perturbation of new ridge nodes, as % of ridge length

        'useDrainageArea': True,          # river width dependant on drainage area: w ~ A^0.4
        'maxRiverWidth': 0.3,             # max river (flat terrain) width, as % of distance between river and ridge

                                          # number of smoothing iterations (larger -> more equalized river)
        'coarseRiverSmoothIters': 4,      # smoothing of the coarse river nodes elevation
        'refinedRiverSmoothIters': 5,     # smoothing of the fine river nodes elevation
        'refinedRiverSmoothPosIters': 1   # smoothing of the fine river nodes position
    }


    # The parameters below have been exposed for extended control,
    # although in practice we did not change them in our tests.
    # They all affect the position and elevation of the river sources,
    # thus the slope between ridges and sources, and indirectly the rest of the river (slightly)
    sourcesParams = {
        # sources initial elevation = ridge elevation - random taken from Normal(mean, std)
        'srcElevRndMean': 50,
        'srcElevRndStd': 20,

        # sources momentum used during smoothing (1 = fixed value)
        # lower momentum values will allow sources to be averaged with subsequent river nodes
        # this usually creates steeper valleys
        'momentumCoarseRiverSourceElevs': 0.5, # for coarse river nodes elev smoothing
        'momentumRiverSourceElev': 0.75,       # for fine river nodes elev smoothing
        'momentumRiverSourceCoords': 0.7,      # for fine river nodes pos smoothing

        # uncomment and tune this distance if rivers are missing between nearby ridges
        #'virtualRidgePointsDist': 3.0     # in terrain units (km)
    }
    
    coordinates_elevation = [peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree]
    parameters = [reconsParams,sourcesParams]
    
    return coordinates_elevation, terrainSize, poissonSamples, parameters

def generate_poisson_samples(terrainSize):
    # Poisson samples precomputation, takes a while.
    # We store them in a file for reuse (optional)
    reusePoissonIfAvailable = True
    poissonFileName = 'poisson_%d_%d.npy'%(terrainSize[0], terrainSize[1])

    if reusePoissonIfAvailable and os.path.exists(poissonFileName):
        poissonSamples = np.load(poissonFileName)
    else:
        # We usually set poissonRadius = 1.5*refineDistance (see later in reconsParams)
        poissonRadius  = 0.18 # km

        poissonSamples = PoissonDisc(width=terrainSize[0], height=terrainSize[1], r=poissonRadius, k=15).sample()
        poissonSamples = np.array([[s[0], s[1]] for s in poissonSamples])

        if reusePoissonIfAvailable:
            np.save(poissonFileName, poissonSamples)
    return poissonSamples

def out_of_bounds_check(peakCoords,saddleCoords,terrainSize):
    out_of_bounds_1 = np.any(peakCoords < 0)
    out_of_bounds_2 = np.any(peakCoords > np.array(terrainSize))
    out_of_bounds_3 = np.any(saddleCoords < 0)
    out_of_bounds_4 = np.any(saddleCoords > np.array(terrainSize))
    
    out_of_bounds_l = [out_of_bounds_1,out_of_bounds_2,out_of_bounds_3,out_of_bounds_4]
    
    if any(out_of_bounds_l):        
        s = f"""WARNING: there are coordinates out of defined terrain size')
        {('Peaks', peakCoords.min(), peakCoords.max())}
        {('Saddles', saddleCoords.min(), saddleCoords.max())}
        """
        raise ValueError(s)
  
## Debug visualizations

def voronoi_and_river_lines(dimensions,saddlePeaks,peakCoords,saddleCoords,debugInfo):
    print("what")
    # change these to crop a portion of the terrain, e.g. for figures
    [xmin,ymin,xmax,ymax] = dimensions
    
    # Voronoi cells and river lines
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    

    # ridgelines from divide tree
    for i in range(saddlePeaks.shape[0]):
        p1 = peakCoords[saddlePeaks[i,0]]
        p2 = peakCoords[saddlePeaks[i,1]]
        ps = saddleCoords[i]
        ax.plot([p1[0], ps[0]], [p1[1], ps[1]], color='orange', linewidth=3, zorder=1)
        ax.plot([p2[0], ps[0]], [p2[1], ps[1]], color='orange', linewidth=3, zorder=1)

    # voronoi cells
    voronoiCells = debugInfo['voronoiRegions']
    voronoiVerts = debugInfo['voronoiVerts']
    for ip,poly in enumerate(voronoiCells):
        for i,p in enumerate(poly):
            p1 = voronoiVerts[p]
            p2 = voronoiVerts[poly[(i+1)%len(poly)]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=1)

    # river lines
    riverLines = debugInfo['coarseRiverLines']
    for line in riverLines:
        for i,p in enumerate(line.coords):
            p1 = p
            p2 = line.coords[(i+1)%len(line.coords)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='steelblue', linewidth=2)


    plt.xlim(xmin-10, xmax+10)
    plt.ylim(ymin-10, ymax+10)
    
    #ax.set_axis_off()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.axis('off')
    #plt.savefig('voronoi.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    a=1

def coarse_river_network(dimensions,saddlePeaks,peakCoords,saddleCoords,debugInfo):
    # coarse river network
    
    [xmin,ymin,xmax,ymax] = dimensions
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)


    # ridgelines from divide tree
    for i in range(saddlePeaks.shape[0]):
        p1 = peakCoords[saddlePeaks[i,0]]
        p2 = peakCoords[saddlePeaks[i,1]]
        ps = saddleCoords[i]
        ax.plot([p1[0], ps[0]], [p1[1], ps[1]], color='orange', linewidth=3, zorder=1)
        ax.plot([p2[0], ps[0]], [p2[1], ps[1]], color='orange', linewidth=3, zorder=1)

    # river flow and drainage area
    voronoiVerts = debugInfo['voronoiVerts']
    riverDrainArea = debugInfo['coarseRiverDrainArea']
    riverFlowTo = debugInfo['coarseRiverFlowTo']
    for rfrom,rto in enumerate(riverFlowTo):
        if rto >= 0:
            p1 = voronoiVerts[rfrom,:]
            p2 = voronoiVerts[rto,:]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='steelblue', linewidth=0.15*riverDrainArea[rto]**0.4)
            #ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color='blue', linewidth=0.01, length_includes_head=True)

    # river sources
    riverSources = debugInfo['coarseRiverSources']
    riverElevs   = debugInfo['coarseRiverElevs']
    for rs in riverSources:
        if not rs in riverFlowTo:
            p1 = voronoiVerts[rs,:]
            ax.scatter(p1[0], p1[1], marker='o', s=20, color='purple', zorder=3)
            #ax.annotate('%.2f'%(riverElevs[rs]), xy=np.clip(p1, 0, terrainSize), fontsize=5)
            #ax.annotate('%d'%(rs), xy=np.clip(p1, 0, terrainSize), fontsize=5)


    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


    #ax.set_axis_off()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.axis('off')
    #plt.savefig('coarseRivers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    a = 1

def refined_networks(dimensions,meshVerts,debugInfo):
    
    [xmin,ymin,xmax,ymax] = dimensions
    # refined networks

    drawPoisson = False
    drawMesh = False


    fig = plt.figure(figsize=(20,20),frameon=False)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    numRidgeVerts = debugInfo['numRidgeVerts']
    numRiverVerts = debugInfo['numRiverVerts']
    numPoissonVerts = debugInfo['numPoissonVerts']

    # refined ridgelines
    ridgeCoords = meshVerts[:numRidgeVerts,:]
    ridgeSegs   = debugInfo['ridgeSegments']
    for seg in ridgeSegs:
        rfrom,rto = seg
        p1 = ridgeCoords[rfrom,:]
        p2 = ridgeCoords[rto,:]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='orange', linewidth=3, zorder=1)


    # river flow and drainage area
    riverCoords = meshVerts[numRidgeVerts:numRidgeVerts+numRiverVerts,:]
    riverDrainArea = debugInfo['riverDrainArea']
    riverFlowTo = debugInfo['riverFlowTo']
    for rfrom,rto in enumerate(riverFlowTo):
        if rto >= 0:
            p1 = riverCoords[rfrom,:]
            p2 = riverCoords[rto,:]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='steelblue', linewidth=0.15*riverDrainArea[rto]**0.4)
            #ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color='blue', linewidth=0.01, length_includes_head=True)

    # Poisson samples
    poissonCoords = meshVerts[numRidgeVerts+numRiverVerts:numRidgeVerts+numRiverVerts+numPoissonVerts,:]
    if drawPoisson:
        ax.scatter(poissonCoords[:,0], poissonCoords[:,1], marker='o', s=5, color='purple')

    # Mesh triangles
    if drawMesh:
        ax.triplot(meshVerts[:,0], meshVerts[:,1], meshTris, color='black', linewidth=0.2, zorder=5)


    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.savefig('refinedNetworks.png',dpi=300, bbox_inches='tight', pad_inches=0)
    #ax.set_axis_off()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.axis('off')
    #plt.savefig('refinedNetworks.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    a = 1

def poisson_something(dimensions,meshVerts,debugInfo):
    
    [xmin,ymin,xmax,ymax] = dimensions
    
    # Poisson samples closest river and ridge
    
    numRidgeVerts = debugInfo['numRidgeVerts']
    numRiverVerts = debugInfo['numRiverVerts']
    numPoissonVerts = debugInfo['numPoissonVerts']
    ridgeCoords = meshVerts[:numRidgeVerts,:]
    riverCoords = meshVerts[numRidgeVerts:numRidgeVerts+numRiverVerts,:]
    poissonCoords = meshVerts[numRidgeVerts+numRiverVerts:numRidgeVerts+numRiverVerts+numPoissonVerts,:]

    closestRidgeIdx = debugInfo['closestRidgeIdx']
    closestRiverIdx = debugInfo['closestRiverIdx']


    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)


    ax.scatter(ridgeCoords[:,0], ridgeCoords[:,1], marker='o', color='orange', s=10)
    ax.scatter(riverCoords[:,0], riverCoords[:,1], marker='o', color='steelblue', s=10)

    for test in np.random.permutation(numPoissonVerts)[:10]:
        p1 = ridgeCoords[closestRidgeIdx[test]]
        p2 = riverCoords[closestRiverIdx[test]]
        ps = poissonCoords[test]

        ax.plot([p1[0], ps[0]], [p1[1], ps[1]], color='darkgreen', linewidth=3)
        ax.plot([p2[0], ps[0]], [p2[1], ps[1]], color='darkgreen', linewidth=3)
        ax.scatter(ps[0], ps[1], marker='o', color='orange', s=100)
        ax.scatter(p1[0], p1[1], marker='o', color='red', s=100)
        ax.scatter(p2[0], p2[1], marker='o', color='blue', s=100)


    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    

if __name__=="__main__":
    mesh_main()
