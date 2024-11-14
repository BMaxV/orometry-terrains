import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from utils import coords
from utils import noise
from utils import metrics
from utils import distributions 

from analysis import peaksdata
from synthesis import divtree_synthesis

def base_setup():
    #get_ipython().run_line_magic('matplotlib', 'inline')
    matplotlib.rcParams['image.cmap'] = 'terrain'

    np.random.seed(42)

def define_constants():
    constants = {
        "promEpsilon"   : 30,   # m,  minimum prominence threshold in the analysis
        "diskRadius"    : 30 ,  # km, used for the analysis to normalize histograms
        "globalMaxElev" : 9000, # m,  any value larger than any other peak elevation, used internally as initialization and undefineds

        "terrainUnitKm"  : 90 , # km, size of terrain
        "km2pixels" : 1000/30 , # 30 m/pixel
        
        # not actually really constant, but let's go with a dictionary for inputs?
        "filterRadius" : 45, # km
        }

    return constants


def target_setup():
    # # Target terrain orometry

    # In[4]:


    #regionName, filterCoords = 'pyrenees', [42.5893, 0.9377] # pyrenees: aiguestortes
    #regionName, filterCoords = 'alps', [45.8325,  7.0]  # mont blanc
    #regionName, filterCoords = 'alps', [44.8742,  6.5]  # ecrins
    #regionName, filterCoords = 'alps', [46.4702, 11.9492] # dolomites
    #regionName, filterCoords = 'alps', [46.0159, 7.74318] # valais
    #regionName, filterCoords = 'sahara', [30.38, 8.69] # sahara dunes
    #regionName, filterCoords = 'andes_chile', [-21.4483, -68.0708] # chile
    #regionName, filterCoords = 'karakoram', [35.8283, 76.3608] # karakoram
    #regionName, filterCoords = 'colorado', [39.0782,-106.6986] # colorado
    #regionName, filterCoords = 'yangshuo', [24.9917, 110.4617] # yangshuo
    #regionName, filterCoords = 'himalaya', [28.7150, 84.2000] # himalaya: annapurna
    #regionName, filterCoords = 'himalaya', [27.8575, 86.8267] # himalaya: everest
    #regionName, filterCoords = 'norway', [62.1167, 6.8075] # norway
    #regionName, filterCoords = 'alaska', [62.9500, -151.0908] # alaska
    #regionName, filterCoords = 'patagonia', [-50.8925, -73.1533] # patagonia
    #regionName, filterCoords = 'andes_aconcagua', [-32.6533, -70.0108] # aconcagua
    regionName, filterCoords = 'andes_peru', [-9.0874, -77.5737] # huascaran
    #regionName, filterCoords = 'rockies', [50.8003, -116.29517] # canadian rockies
    #regionName, filterCoords = 'appalachians', [35.3855, -83.2380] # appalachians
    #regionName, filterCoords = 'highlands', [56.9667, -3.5917] # highlands

    peaksFilename = 'data/regionPeaks/%s.csv' % regionName


    return regionName, filterCoords, peaksFilename

def synthesis_filter(peaksFilename,filterCoords,constants):
    

    filterRadius = constants["filterRadius"]

    filterHWidth = [coords.km2deg(filterRadius), coords.km2deg(filterRadius, filterCoords[0])]
    print(filterCoords[0] - filterHWidth[0], filterCoords[0] + filterHWidth[0],
          filterCoords[1] - filterHWidth[1], filterCoords[1] + filterHWidth[1])

    # read peaks file and filter region of interest
    df = pd.read_csv(peaksFilename)

    filterHWidth = [coords.km2deg(filterRadius), coords.km2deg(filterRadius, filterCoords[0])]
    filat = np.logical_and(df['latitude']  > filterCoords[0] - filterHWidth[0],
                           df['latitude'] < filterCoords[0] + filterHWidth[0])
    filon = np.logical_and(df['longitude'] > filterCoords[1] - filterHWidth[1],
                           df['longitude'] < filterCoords[1] + filterHWidth[1])
    df = df[np.logical_and(filat, filon)]

    print('Peaks:', df.shape[0])
    df = peaksdata.addExtraColumns(df)

    return df


def compute_distribution(df):
    # compute distributions

    distributions_dict = computeDistributions(df, diskRadius)
    return distributions_dict


def visualize_peaks(df):
    # Visualize the peaks
    fig = plt.figure(figsize=(5,5))
    plt.scatter(df['longitude'], df['latitude'], marker='^',
                    s=20*df['elev'].values/df['elev'].values.max(), c=df['elev'].values/df['elev'].values.max())

def visualize_distributions(distributions_dict,img_control_vars):
    # Visualize distributions
    barColor = img_control_vars["barColor"]
    edgeColor = img_control_vars["edgeColor"]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(231)
    h = ax.bar(distributions_dict['elevation']['x'],
               distributions_dict['elevation']['hist'],
               width=np.diff(distributions_dict['elevation']['bins']), color=barColor, edgecolor=edgeColor)

    ax = fig.add_subplot(232)
    h = ax.bar(distributions_dict['prominence']['x'],
               distributions_dict['prominence']['hist'],
               width=np.diff(distributions_dict['prominence']['bins']), color=barColor, edgecolor=edgeColor)

    ax = fig.add_subplot(233)
    h = ax.bar(distributions_dict['dominance']['x'],
               distributions_dict['dominance']['hist'],
               width=np.diff(distributions_dict['dominance']['bins']), color=barColor, edgecolor=edgeColor)

    ax = fig.add_subplot(234)
    h = ax.bar(distributions_dict['isolation']['x'],
               distributions_dict['isolation']['hist'],
               width=np.diff(distributions_dict['isolation']['bins']), color=barColor, edgecolor=edgeColor)

    ax = fig.add_subplot(235, polar=True)
    ax.set_yticklabels([])
    h = ax.bar(np.radians(distributions_dict['isolDir']['x']),
               distributions_dict['isolDir']['hist'],
               width=np.diff(np.radians(distributions_dict['isolDir']['bins'])), color=barColor, edgecolor=edgeColor)

    ax = fig.add_subplot(236, polar=True)
    ax.set_yticklabels([])
    h = ax.bar(np.radians(distributions_dict['saddleDir']['x']),
               distributions_dict['saddleDir']['hist'],
               width=np.diff(np.radians(distributions_dict['saddleDir']['bins'])), color=barColor, edgecolor=edgeColor)


def control(constants):
    # # Control

    # control images
    pathControlDEM = 'input/user_elevs.png'
    pathControlDensity = 'input/user_probs.png'

    imgControlDEM = np.asarray(Image.open(pathControlDEM)).astype(float)
    if len(imgControlDEM.shape) > 2:
        imgControlDEM = imgControlDEM[:,:,0].squeeze()
    imgControlDEM = np.fliplr(imgControlDEM.T)
    imgControlDEM /= imgControlDEM.max()
    shapeImg = imgControlDEM.shape

    imgControlDensity = np.asarray(Image.open(pathControlDensity)).astype(float)/255
    if len(imgControlDensity.shape) > 2:
        imgControlDensity = imgControlDensity[:,:,0].squeeze()
    imgControlDensity = np.fliplr(imgControlDensity.T)
    imgControlDensity /= imgControlDensity.max()
    shapeImg = imgControlDensity.shape


    # account for non-square terrains
    terrainAspect = np.array([1.0, 1.0])
    if shapeImg[0] > shapeImg[1]:
        terrainAspect[0] = shapeImg[0]/shapeImg[1]
    else:
        terrainAspect[1] = shapeImg[1]/shapeImg[0]

    terrainSize = terrainAspect * constants["terrainUnitKm"]
    
    img_control_vars = {"imgControlDensity":imgControlDensity,
                        "imgControlDEM":imgControlDEM, 
                        "shapeImg":shapeImg,
                        "terrainAspect":terrainAspect,
                        "terrainSize":terrainSize,
                        "barColor":(216/255, 226/255, 238/255, 1.0),
                        "edgeColor":(137/255, 151/255, 168/255, 1.0),
                        }
    return img_control_vars

def enter_predfined():
    
    predefined = {
    # predefined peaks and saddles
    "fixPeakCoords"   : np.array([[0.65,0.2], [0.625,0.22], [0.69,0.19], [0.9,0.92]]),
    "fixPeakElevs"    : np.array([4500, 4200, 4050, 3000]),
    "fixSaddleCoords" : np.array([[0.63, 0.21], [0.68, 0.195]]),
    "fixSaddlePeaks"  : np.array([[0, 1], [0, 2]]),

    # do we want these predefined peaks to have a particular prominence/dominance value or range?
    "peakRangeProm" : np.array([[4500,4500], [500,2000], [500,2000], [0,4000]]),
    
    }
    
    prom = predefined["peakRangeProm"]
    elev = predefined["fixPeakElevs"][:,np.newaxis,]
    
    dom = prom / elev
    
    predefined["peakRangeDom"] = dom
    
    return predefined
    
    
def create_probability_density_maps(img_control_vars):
    # create probability (density) maps
    density = img_control_vars["imgControlDensity"]
    
    regular_min_diff = (density - density.min())
    extreme_diff = (density.max() - density.min())
    
    probMap = regular_min_diff / extreme_diff
    probMapSaddles = regular_min_diff / extreme_diff
    
    origMap = img_control_vars["imgControlDEM"]
    
    return probMap, probMapSaddles, origMap


def forbid_interference(predefined,img_control_vars,probMap):
    # Optional: we might be interested in forbidding peak placement close to the given predefined ridges
    dfRidges, imgRidges = divtree_synthesis.getRidgesDF(
                                    predefined["fixPeakCoords"],
                                    predefined["fixSaddleCoords"], 
                                    predefined["fixSaddlePeaks"], 
                                    img_control_vars["shapeImg"],
                                    img_control_vars["terrainAspect"],
                                    ridgesWidth=1,
                                    normalized=False)
                                     
    probMap[dfRidges <= 2] = 0

def add_noise(img_control_vars):
    # add some noise to elevation map
    dem = img_control_vars["imgControlDEM"]
    
    weightNoise = 0.05
    my_noise = noise.getNoiseTexture(dem.shape, seed=4.2, scale=4)
    elevMap = (1 - weightNoise)*dem + weightNoise*my_noise
    return elevMap

def map_to_histogram(probMap,elevMap,origMap,distributions_dict,df):
    """
    # histogram match the elevation map
    # on values = 0, we will put random elevations in order to not modify the stats of the rest
    # note the values on them are not important as the probability of placing a peak there will always be 0
    """
    elevMap[probMap > 0] = distributions.equalize(elevMap[probMap > 0], numBins=1024)
    elevMap[probMap <= 0] = np.random.uniform(0, 1, size=probMap[probMap <= 0].shape)
    elevMap = distributions.mapToPDF(elevMap.flatten(),
                       distributions_dict['elevation']['hist'],
                       distributions_dict['elevation']['bins']).reshape(elevMap.shape)
    elevMap[probMap <= 0] = df['saddleElev'].min()


    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(131)
    ax.imshow(np.flipud(origMap.T))
    ax = fig.add_subplot(132)
    ax.imshow(np.flipud(elevMap.T))
    ax = fig.add_subplot(133)
    ax.imshow(np.flipud(probMap.T))
    
    return elevMap


def synthesis_main():
    base_setup()
    constants = define_constants()
    img_control_vars = control(constants)
    
    regionName, filterCoords, peaksFilename = target_setup()

    df = synthesis_filter(peaksFilename, filterCoords, constants)
    
    distributions_dict = peaksdata.computeDistributions(df, constants["diskRadius"])
    
    visualize_peaks(df)
    visualize_distributions(distributions_dict,img_control_vars)

    predefined = enter_predfined() # THIS IS ACTUALLY THE INPUT ?
    
    probMap, probMapSaddles, origMap = create_probability_density_maps(img_control_vars )
    
    forbid_interference(predefined,img_control_vars,probMap)
    
    elevMap = add_noise(img_control_vars)
    elevMap = map_to_histogram(probMap,elevMap,origMap,distributions_dict,df)

    #MAIN PROGRAM
    promGroups, totalNumPeaks = get_stats(probMap,constants,df)
    comp_distributions = compute_distribution_for_prominence_group(promGroups,totalNumPeaks,df,constants)
    
    # not exactly fixed data, is it?
    synthParams, fixedData = get_param_fixed_dicts(img_control_vars, predefined)
    elev_dict, RidgeTree, debugInfo = synthesis(distributions_dict,comp_distributions,probMap,probMapSaddles,elevMap, fixedData, synthParams)
    
    draw_divide_tree(elev_dict,img_control_vars) # what's that then?
    peaksfile(regionName, elev_dict)

    make_divide_tree(regionName, elev_dict, constants, RidgeTree)
    #divide_tree_image(img_control_vars) # huh? another one?

    debug_visualizations(debugInfo,img_control_vars,distributions_dict,elev_dict)
    
def get_stats(probMap,constants,df):
    # total number of peaks to synthesize on the terrain, based on the density from the given region
    densityFactor = (probMap > 0).sum()/np.prod(probMap.shape)
    
    u_km = constants["terrainUnitKm"]
    r_f = constants["filterRadius"]
    
    scalingFactor = (u_km/(2 * r_f))**2
    totalNumPeaks = np.round(densityFactor * scalingFactor * df.shape[0])

    print('Density factor (average density in probMap):', '%.2f'%densityFactor)
    print('Scaling factor (area ratio out / analysis): ', scalingFactor)
    print('NUM PEAKS:', totalNumPeaks)

    # get the prominence thresholds that divide successively in halfs the peaks in the dataset
    pros = sorted(df['prom'])
    steps = [8/15, 12/15, 14/15]
    for s in steps:
        print(int(pros[int(s*len(pros))]))

    # each pair of consecutive values defines the max-min prominence range for each multi-pass placement step
    promGroups = [constants["globalMaxElev"], 260, 140, 75, 0]
    return promGroups, totalNumPeaks

def compute_distribution_for_prominence_group(promGroups,totalNumPeaks,df,constants):
    # compute the conditioned distributions for each prominence group
    binDistributions = []
    accDistributions = []
    promStepNumPeaks = []
    promGroupsLimits = []
    for gi in range(len(promGroups) - 1):

        maxProm = promGroups[gi]
        minProm = promGroups[gi+1]
        
        # this seems like a very complicated way to just set 
        # df[maxProm, minProm] no?
        promGroupDF = df[np.logical_and(df['prom'] >= minProm, df['prom'] < maxProm)]
        promAccumDF = df[df['prom'] >= minProm]

        promStepNumPeaks.append(int(np.round(totalNumPeaks * (promGroupDF.shape[0]/df.shape[0]))))
        binDistributions.append(peaksdata.computeDistributions(promGroupDF, constants["diskRadius"]))
        accDistributions.append(peaksdata.computeDistributions(promAccumDF, constants["diskRadius"]))
        promGroupsLimits.append((minProm, maxProm))
    
    comp_distributions = {
        "promStepNumPeaks":promStepNumPeaks,
        "binDistributions":binDistributions,
        "accDistributions":accDistributions,
        "promGroupsLimits":promGroupsLimits,
    }
    
    return comp_distributions
    

def get_param_fixed_dicts(img_control_vars,predefined):
    
    # parameters dictionary
    synthParams = {

        # Global parameters
        'promEpsilon': 30,          # m, minimum prominence used in the peak analysis
        'globalMaxElev': 9000,      # m, maximum elevation, any value larger than highest peak will do
        'terrainSize': img_control_vars["terrainSize"], # dimensions of the domain

        # Range (in % of elevation map range) used to prefilter the candidate positions for each peak dart throwing.
        # This accelerates peak placement but reduces randomness of generated terrains,
        # Value of 0 means that only those positions in the elevation map that have the same elevation as the peak
        # we are trying to place are candidates for the dart throwing.
        # Using a value of 1 disables the optimization, as all positions in the map will be candidates.
        'elevRangeFilter': 0.5,

        # Maximum number of positions to test during each peak placement step. If reached, the position that scored
        # the highest probability will be selected as final location.
        'maxPeakTrials': 100,

        # Exponent used in the cost of each Delaunay edge when constructing the Divide Tree,
        # which serves as a balancing factor between closest edges (0) or highest edges (1, 2)
        'delaunayRidgeExp': 1.0,

        # Update probability map after each multi-pass placement to take into account the distance to ridges?
        'updateProbMap': True,

        # used to control the extent of the ridges and valleys during multi-pass peak placement
        # larger values lead to wider empty valleys, smaller values cover more uniformly the domain but less structured
        'valleyFactor': 1,  # typically something between 0.5 to 1.5

        # Number of iterations of prominence/dominance optimization. Usually 3-4 are enough, check debug info later
        'numHistogramIters': 5
    }


    
    var_1 = predefined["fixPeakCoords"]*img_control_vars["terrainSize"]
    var_2 = predefined["fixPeakElevs"][:,np.newaxis]
    # fixed data dictionary
    fixedData = {
        'fixedPeaks': np.hstack([var_1,var_2]),
        'peakRangeProm': predefined["peakRangeProm"],
        'peakRangeDom': predefined["peakRangeDom"],
        'fixedSaddles': np.empty((0,3)),
        'fixedSaddlesPeaks': np.empty((0,2), dtype=int)
    }
    
    # code snippet without fixed data, still need to pass the vectors!
    #fixedData = {
    #    'fixedPeaks': np.empty((0,3)),
    #    'peakRangeProm': np.empty((0,2)),
    #    'peakRangeDom': np.empty((0,2)),
    #    'fixedSaddles': np.empty((0,3)),
    #    'fixedSaddlesPeaks': np.empty((0,2), dtype=int)
    #}

    return synthParams, fixedData


def synthesis(distributions_dict,comp_distributions,probMap,probMapSaddles,elevMap, fixedData, synthParams):
    # SYNTHESIS
    np.random.seed(42) #repeated definition?
    
    bin_dist = comp_distributions["binDistributions"]
    acc_dist = comp_distributions["accDistributions"]
    prom_g_l = comp_distributions["promGroupsLimits"]
    prom_snp = comp_distributions["promStepNumPeaks"]
    
    out = divtree_synthesis.synthDivideTree(distributions_dict, bin_dist, acc_dist, prom_g_l, prom_snp,
                        probMap, probMapSaddles, elevMap, fixedData, synthParams)
    
    peakCoords, peakElevs, saddleCoords, saddleElevs = out[0:4]
    saddlePeaks, RidgeTree, debugInfo = out[4:]
    
    
    # compute prominences
    peakSaddle, peakParent, peakProms, _ = metrics.computeProminences(RidgeTree, peakElevs, saddleElevs, saddlePeaks)

    # compute dominances
    peakDoms = peakProms / peakElevs

    # compute isolations
    peakIsols, isolCoords = metrics.computeIsolations(peakCoords, peakElevs)

    print('DONE!')
    
    # the machine spirit is angered by this naming.
    elev_dict = {
        "saddleElevs":saddleElevs,
        "saddlePeaks":saddlePeaks,
        "saddleCoords":saddleCoords,
        
        "peakElevs":peakElevs,
        "peakSaddle":peakSaddle,
        "peakCoords":peakCoords,
        "peakProms":peakProms,
        
        "isolCoords":isolCoords,
        "peakIsols":peakIsols,
        "peakDoms":peakDoms,
        
        }
    
    return elev_dict, RidgeTree, debugInfo
    
def draw_divide_tree(elev_dict,img_control_vars,):
    """
    draw resulting divide tree,
    peak size and color represents elevation
    saddle size represents the prominence of 
    the corresponding peak j: KeySaddle(P_j) = S_i
    """
    
    s2p = np.full(elev_dict["saddleElevs"].shape, -1)
    for i in range(elev_dict["peakElevs"].size):
        s2p[elev_dict["peakSaddle"][i]] = i
    
    base_size = np.array(img_control_vars["terrainAspect"])
    
    fig = plt.figure(figsize=(16*base_size))
    ax = fig.add_subplot(111)
    
    peak_c = elev_dict["peakCoords"]
    peakElevs = elev_dict["peakElevs"]
    
    
    style = {
            "linewidths":0.75,
            "edgecolors":(0,0,0,1),
            "marker":'^',
            "s": 150*peakElevs/peakElevs.max(),#marker size
            "c": peakElevs/peakElevs.max(), #marker colors}
            "zorder":2,
            }
    
    ax.scatter(peak_c[:,0], peak_c[:,1], **style)
    
    # at saddle coordinates, color according to peak prominence. 
    
    peakProms = elev_dict["peakProms"]
    saddle_c = elev_dict["saddleCoords"]
    
    style = {
        "marker":'o',
        "color":"r",
        "s":200*peakProms[s2p]/peakProms.max(),
        "zorder":2,
        }
    
    ax.scatter(saddle_c[:,0], saddle_c[:,1], **style)
    
    style = {
        "color":'orange',
        "linewidth":2,
        "zorder":1,
        }
    
    saddleElevs = elev_dict["saddleElevs"]
    saddlePeaks = elev_dict["saddlePeaks"]
    
    for i in range(saddleElevs.size):
        p1 = peak_c[saddlePeaks[i,0]]
        p2 = peak_c[saddlePeaks[i,1]]
        ps = saddle_c[i]
        ax.plot([p1[0], ps[0]], [p1[1], ps[1]], **style)
        ax.plot([p2[0], ps[0]], [p2[1], ps[1]], **style)

    if False: # annotate
        for i in range(peakElevs.size):
            s = peakSaddle[i]
            if s < 0:
                print('No saddle for ', i)
                continue
            #ax.annotate('%d: %d (%d)'%(i, int(peakElevs[i]), int(peakProms[i])), xy=peakCoords[i,:], fontsize=2)
            #ax.annotate('%d (%d)'%(saddleElevs[s], s), xy=saddleCoords[s], fontsize=2)

            ax.annotate('P %d'%(i), xy=peakCoords[i,:], fontsize=4)
            ax.annotate('S %d'%(s), xy=saddleCoords[s], fontsize=4)


    plt.xlim(0, img_control_vars["terrainSize"][0])
    plt.ylim(0, img_control_vars["terrainSize"][1])
    
    # save the tree in pdf (better for debugging)
    if False:
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.axis('off')
        fig.savefig('dividetree.pdf', dpi=100, bbox_inches='tight', pad_inches=0)


def peaksfile(regionName,elev_dict, outName = 'synthesis'):
    # ### Peaks file

    peakElevs = elev_dict["peakElevs"]
    peakCoords = elev_dict["peakCoords"]
    
    saddleElevs = elev_dict["saddleElevs"]
    saddleCoords = elev_dict["saddleCoords"]
    
    isolCoords = elev_dict["isolCoords"]
    
    peakIsols = elev_dict["peakIsols"]
    peakSaddle = elev_dict["peakSaddle"]
    
    

    # In[ ]:
    filename ='%s_%s.csv' % (outName, regionName)
    with open(filename, 'w') as fout:
        fout.write('latitude,longitude,elevation in feet,key saddle latitude,key saddle longitude,' +
                   'prominence in feet,isolation latitude,isolation longitude,isolation in km\n')

        for pi in range(peakElevs.size):

            cPeak = coords.km2deg(peakCoords[pi])
            hPeak = coords.m2feet(peakElevs[pi])
            si = peakSaddle[pi]

            if pi == peakElevs.argmax():
                cSadd = [0, 0]
                hSadd = 0
                cIsol = [0, 0]
                isolation = -1
                
            else:
                cSadd = coords.km2deg(saddleCoords[si])
                hSadd = coords.m2feet(saddleElevs[si])
                cIsol = coords.km2deg(isolCoords[pi,:])
                isolation = peakIsols[pi]

            fout.write('%.4f,%.4f,%d,%.4f,%.4f,%d,%.4f,%.4f,%4f\n' % (cPeak[0], cPeak[1], int(hPeak + 0.5),
                                                                      cSadd[0], cSadd[1], int((hPeak - hSadd) + 0.5),
                                                                      cIsol[0], cIsol[1], isolation))


def make_divide_tree(regionName, elev_dict, constants, RidgeTree, outName = 'synthesis'):
    # ### Divide tree
    
    
    peakElevs = elev_dict["peakElevs"]
    peakCoords = elev_dict["peakCoords"]
    
    saddleElevs = elev_dict["saddleElevs"]
    saddleCoords = elev_dict["saddleCoords"]
    
    isolCoords = elev_dict["isolCoords"]
    
    peakIsols = elev_dict["peakIsols"]
    peakSaddle = elev_dict["peakSaddle"]
    
    peakSaddle = elev_dict["peakSaddle"]
    
    
    
    filename = '%s_%s_dividetree.txt' % (outName, regionName)
    
    
    km2pixels = constants["km2pixels"]
    
    with open(filename, 'w') as fout:

        fout.write('Peaks %d\n' % peakElevs.size)
        for i in range(peakElevs.size):
            fout.write('%d %.6f %.6f %d %d %d\n'%(i, coords.km2deg(peakCoords[i,0]), coords.km2deg(peakCoords[i,1]),
                                                  int(coords.m2feet(peakElevs[i]) + 0.5),
                                                  int(peakCoords[i,0]*km2pixels), int(peakCoords[i,1]*km2pixels)))

        fout.write('PromSaddles %d\n' % saddleElevs.size)
        for i in range(saddleElevs.size):
            fout.write('%d %.6f %.6f %d %d %d\n'%(i, coords.km2deg(saddleCoords[i,0]), coords.km2deg(saddleCoords[i,1]),
                                                  int(coords.m2feet(saddleElevs[i]) + 0.5),
                                                  int(saddleCoords[i,0]*km2pixels), int(saddleCoords[i,1]*km2pixels)))

        fout.write('BasinSaddles 0\n')
        fout.write('Runoffs 0\n')

        edges = []
        for i in range(peakElevs.size):
            for j in range(peakElevs.size):
                if RidgeTree[i,j] >= 0:
                    edges.append((i+1, j+1, RidgeTree[i,j]+1))

        fout.write('Edges %d\n' % len(edges))
        for e in edges:
            fout.write('%d %d %d\n' % (e[0], e[1], e[2]))

        fout.write('RunoffEdges 0\n')


def divide_tree_image(img_control_vars):
    """this one seems redundant?"""
    # ### Divide tree image

    colorTree = True
    if colorTree:
        fig = plt.figure(figsize=(16*np.array(img_control_vars["terrainAspect"])))
        ax = fig.add_subplot(111)
        
        ax.scatter(peakCoords  [:,0], peakCoords  [:,1], marker='^', zorder=2,
                   s=200*peakElevs/peakElevs.max(), c=peakElevs/peakElevs.max(), linewidths=0.75, edgecolors=(0,0,0,1))
        
        ax.scatter(saddleCoords[:,0], saddleCoords[:,1], marker='o', color='r', s=5, zorder=2)
        
        for i in range(saddleElevs.size):
            p1 = peakCoords[saddlePeaks[i,0]]
            p2 = peakCoords[saddlePeaks[i,1]]
            ps = saddleCoords[i]
            ax.plot([p1[0], ps[0]], [p1[1], ps[1]], color='orange', linewidth=2, zorder=1)
            ax.plot([p2[0], ps[0]], [p2[1], ps[1]], color='orange', linewidth=2, zorder=1)
        plt.xlim(0, terrainSize[0])
        plt.ylim(0, terrainSize[1])
        fig.savefig('%s_%s_dividetree.png' % (outName, regionName), dpi=100, bbox_inches='tight', pad_inches=0)

    else:
        img = np.zeros((int(terrainSize[0]*km2pixels), int(terrainSize[1]*km2pixels), 3))

        peakImgCoords   = np.round(peakCoords*km2pixels).astype(int)
        saddleImgCoords = np.round(saddleCoords*km2pixels).astype(int)

        # peaks
        for i in range(peakElevs.size):
            cv.circle(img, (peakImgCoords[i,0], peakImgCoords[i,1]), 6, color=(0,0,1), thickness=-1)

        # saddles
        for i in range(saddleElevs.size):
            cv.circle(img, (saddleImgCoords[i,0], saddleImgCoords[i,1]), 3, color=(0,1,0), thickness=-1)

        # ridges
        for i in range(saddleElevs.size):
            if saddlePeaks[i,0] < 0 or saddlePeaks[i,1] < 0:
                continue
            p1 = peakImgCoords[saddlePeaks[i,0]]
            p2 = peakImgCoords[saddlePeaks[i,1]]
            ps = saddleImgCoords[i]
            cv.line(img, tuple(p1), tuple(ps), color=(0, 0.5, 1), thickness=2)
            cv.line(img, tuple(p2), tuple(ps), color=(0, 0.5, 1), thickness=2)

        img = cv.flip(img, 0)
        _ = cv.imwrite('%s_%s_dividetree.png' % (outName, regionName), 255*img)

def debug_visualizations(debugInfo,img_control_vars,distributions_dict,elev_dict):
    # # Debug visualizations
    draw_evolution(debugInfo,img_control_vars)
    convergence(debugInfo)
    histograms(distributions_dict,elev_dict,img_control_vars)

def draw_evolution(debugInfo,img_control_vars):
    # evolution of the divide tree during the multi-pass placement
    fig = plt.figure(figsize=(20,20))

    for i,tdata in enumerate(debugInfo['stepDivtrees']):

        si_peakCoords, si_peakElevs, si_saddleCoords, si_saddlePeaks = tdata

        ax = fig.add_subplot(2, 2, i+1)
        ax.scatter(si_peakCoords[:,0], si_peakCoords[:,1], marker='^', s=50*si_peakElevs/si_peakElevs.max(), color='r')
        ax.scatter(si_saddleCoords[:,0], si_saddleCoords[:,1], marker='o', s=2, color='g')
        for i in range(si_saddlePeaks.shape[0]):
            p1 = si_peakCoords[si_saddlePeaks[i,0]]
            p2 = si_peakCoords[si_saddlePeaks[i,1]]
            ps = si_saddleCoords[i]
            ax.plot([p1[0], ps[0]], [p1[1], ps[1]], color='b', linewidth=0.5)
            ax.plot([p2[0], ps[0]], [p2[1], ps[1]], color='b', linewidth=0.5)

        plt.xlim(0, img_control_vars["terrainSize"][0])
        plt.ylim(0, img_control_vars["terrainSize"][1])


    # probability map on each placement step
    fig = plt.figure(figsize=(20,5))
    for i,iprobmap in enumerate(debugInfo['stepProbMaps']):
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(np.flipud(iprobmap.T))


def convergence(debugInfo):

    # convergence of prominence and dominance towards target distribution
    # each curve represents the sum of absolute bin differences between current and target histograms
    domDifferences = debugInfo['domDifferences']
    promDifferences = debugInfo['promDifferences']

    plt.figure()
    plt.plot(domDifferences, c='b')
    plt.plot(promDifferences, c='r')
    plt.plot(np.array(promDifferences) + np.array(domDifferences), c='g')

def printHistogramsDistances(hbins, hReal, hSynth):
    hdiff = np.abs(hReal - hSynth)
    print('Max', np.max(hdiff), 'Sum', np.sum(hdiff), 'Avg', np.mean(hdiff))
    print('EMD', np.diff(hbins)[0]*np.abs(np.cumsum(hReal) - np.cumsum(hSynth)).sum())

def histogramsComparison(distribution, synthesisValues, img_control_vars):
    
    barColor = img_control_vars["barColor"]
    edgeColor = img_control_vars["edgeColor"]
    
    hbins  = distribution['bins']
    hmids  = distribution['x']
    hReal  = distribution['hist']
    hSynth = peaksdata.histogramFromBins(synthesisValues, hbins, frequencies=False)
    hNorm  = np.round(synthesisValues.size * hReal/hReal.sum())

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(131)
    ax.bar (hmids, hSynth, width=np.diff(hbins), color=barColor, edgecolor=edgeColor)
    ax.plot(hmids, hNorm, color='r')

    ax = fig.add_subplot(132)
    ax.bar (hmids, hNorm, width=np.diff(hbins), color='g')
    ax.plot(hmids, hNorm, color='r')

    printHistogramsDistances(hbins, hReal/hReal.sum(), hSynth/hSynth.sum())

    print('Per bin differences (synthesis - target)')
    print(hSynth - hNorm)

def histograms(distributions_dict,elev_dict,img_control_vars):
    
    peakElevs = elev_dict["peakElevs"]
    peakProms = elev_dict["peakProms"]
    peakDoms = elev_dict["peakDoms"]
    peakIsols = elev_dict["peakIsols"]
    
    
    histogramsComparison(distributions_dict['elevation'], peakElevs, img_control_vars)
    histogramsComparison(distributions_dict['prominence'], peakProms, img_control_vars)
    histogramsComparison(distributions_dict['dominance'], peakDoms, img_control_vars)

    # note that it is actually ok for our computed isolations to be slightly larger than target histogram
    # we are computing the approximate peak-to-peak isolation, instead of peak-to-ground
    # if we compute the real isolations after reconstructing the DEM, they will be smaller
    histogramsComparison(distributions_dict['isolation'], peakIsols, img_control_vars)

if __name__=="__main__":
    synthesis_main()

