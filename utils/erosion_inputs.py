import numpy as np

import cv2 
# it looks like you're using cv2 just to draw some circles, lines and 
# outputting some pictures? Surely there is a better way.

# causes a major interaction problem with matplotlib.
# apparently one of the fixes is to import a headless version?
# you want opencv-python-headless


def heightfield_delaunay(coords, elevs, terrainSize, hfsize):
    
    """
    Function provided as debug visualization of the DEM
    Introduces small errors due to pixel resolution and elevation discretization.
    Also, we use scipy delaunay because it contains useful functions that speed up the code,
    but unlike Triangle library it's unconstrained and might omit some (small) ridge segments thus creating saddles/peaks.
    However, since we already sampled points over the whole domain and we include the Steiner points, should be almost equal.
    For best results, directly rasterize the geometry mesh obtained above.

    https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
    """
    
    
    # pixel coords
    x = np.linspace(0, 1, hfsize[0] + 1)[:-1] + 0.5/hfsize[0]
    y = np.linspace(0, 1, hfsize[1] + 1)[:-1] + 0.5/hfsize[1]
    xv, yv = np.meshgrid(x, y)
    pixcoords = np.array([xv.flatten(), yv.flatten()]).T

    # compute Delaunay, and find the triangle containing each target (-1 if not found)
    pointCoords = np.concatenate([coords/terrainSize, np.array([[0, 0], [0, 1], [1, 1], [0, 0]])])
    pointElevs  = np.concatenate([elevs,  np.array([0, 0, 0, 0])])
    pointElevs  = pointElevs[:, np.newaxis]
    delaunayTri = Delaunay(pointCoords)
    triangles   = delaunayTri.find_simplex(pixcoords)

    # compute barycentric coordinates
    X = delaunayTri.transform[triangles,:2]
    Y = pixcoords - delaunayTri.transform[triangles,2]
    b = np.einsum('...ij,...j->...i', X, Y) # multiply and sum last dimension of X with last dimension of Y
    bcoords = np.c_[b, 1 - b.sum(axis=1)]

    # interpolate elevations
    ielevs = np.einsum('ij,ijk->ik', bcoords, pointElevs[delaunayTri.simplices[triangles]])

    # store result
    pixels = (hfsize * pixcoords).astype(int)
    hfield = np.zeros(hfsize)
    hfield[pixels[:,0], pixels[:,1]] = ielevs.flatten()
    
    # I think this is done alyways and anyway. so... might as well 
    # do that here.
    hfield = np.rot90(hfield)
    
    return hfield
 

def image_settings(terrainSize, pixelMeters = 30):
    # this is some matplotlib settin function, no idea...
    #mpl.rcParams['image.cmap'] = 'terrain'
    hfsize = (np.array(terrainSize)*1000/pixelMeters).astype(int)
    return hfsize

def make_delauney_picture(hf,terrainName):
    """
    this function produces output that would be input for the erosion code, but since the erosion code is not public, 
    
    there is not much point to having it. I will keep it in but if it can be removed, that would be great, because one less depedency (to opencv)
    """
    cv2.imwrite(terrainName + '_dem16.png', np.maximum(10*hf, 0).astype(np.uint16))
    
def distance_fields(hfsize, terrainName,terrainSize,peakCoords,saddleCoords,meshVerts,debugInfo):
    """
    this function produces output that would be input for the erosion code, but since the erosion code is not public, 
    
    there is not much point to having it. I will keep it in but if it can be removed, that would be great, because one less depedency (to opencv)
    """
    # ### Distance fields
    
    # distance fields to use in the erosion and enhancement part of the algorithm
    dfPeaks = np.ones(hfsize)
    for p in peakCoords:
        pi = (p/terrainSize*hfsize).astype(int)
        cv2.circle(dfPeaks, (pi[1], pi[0]), 2, color=0, thickness=-1)
    dfPeaks = cv2.distanceTransform(dfPeaks.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dfPeaks = np.minimum(dfPeaks, 255.0)
    
    dfSaddles = np.ones(hfsize)
    for s in saddleCoords:
        pi = (s/terrainSize*hfsize).astype(int)
        cv2.circle(dfSaddles, (pi[1], pi[0]), 2, color=0, thickness=-1)
    dfSaddles = cv2.distanceTransform(dfSaddles.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dfSaddles = np.minimum(dfSaddles, 255.0)
    
    dfRidges = np.ones(hfsize)
    ridgeCoords = meshVerts[:debugInfo['numRidgeVerts'], :]
    for seg in debugInfo['ridgeSegments']:
        rfrom,rto = seg
        p1 = (ridgeCoords[rfrom,:]/terrainSize*hfsize).astype(int)
        p2 = (ridgeCoords[rto,:]/terrainSize*hfsize).astype(int)
        cv2.line(dfRidges, (p1[1], p1[0]), (p2[1], p2[0]), color=0, thickness=1)
    dfRidges = cv2.distanceTransform(dfRidges.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dfRidges = np.minimum(dfRidges, 255.0)
    
    imgDF = np.rot90(np.dstack([dfRidges, dfSaddles, dfPeaks]))
    cv2.imwrite(terrainName + '_distfield.png', imgDF.astype(np.uint8))
