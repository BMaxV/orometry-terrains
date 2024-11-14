import argparse
import shapefile  # http://github.com/GeospatialPython/pyshp
import shapely    # http://toblerity.org/shapely/manual.html
from shapely.geometry import Point, Polygon, MultiPolygon
#import multiprocessing


def cmd():
    args = get_args()

    output_string = core_without_output(args.shapesFile,args.pointsFile,args.tolerance)
    
    make_output(output_string)
    
def core_without_output(shapes_filename,points_filename,tolerance=0):
    
    pointsfile_text, sf = get_processing_input(shapes_filename,points_filename)
    
    output_string = processing(pointsfile_text,sf,tolerance) 
    
    return output_string

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("shapesFile", help=".shp with the polygons of the regions")
    parser.add_argument("pointsFile", help=".txt file with the points to check")
    parser.add_argument("outFile", help="output file")
    parser.add_argument("--tolerance", help="distance (m) around polygon test", default=0, type=float)
    args = parser.parse_args()
    return args

def get_processing_input(shapes_filename,points_filename):
    sf = shapefile.Reader(shapes_filename)
    with open(shapes_filename) as f:
        sfencoding = f.encoding
        
    with open(points_filename,"r", encoding='utf-8') as f:
        pointsfile_text = f.read()
    return pointsfile_text, sf

    
def make_output(args,output_string):
    with open(args.outFile,"w") as f:
        f.write(output_string)


def local_multi(data_in):
    num_processes = 30
    mypool = multiprocessing.Pool(30)
    #sub_list = pointsfile_lines[counter:counter+batch_length]
    #mypool.apply_async(single_line_check,(
    active_stuff = []
    not_finished = len(active_stuff) > 0
    input_left = (len(data_in) > 0)
    data_out = []
    
    my_function = batch_check
    
    with multiprocessing.Pool(processes=number_of_processes) as my_pool:
        while input_left or not_finished:

            processes_are_idle = len(active_stuff) < number_of_processes
            input_left = len(data_in) > 0

            while processes_are_idle and input_left:
                
                args = data_in.pop(0)  #input_left makes sure this exists
                kwargs = {}
                handle = my_pool.apply_async(my_function, args, kwargs)
                
                active_stuff.append(handle)
                input_left = len(data_in) > 0
                
            for handle in active_stuff:
                if handle.ready():
                    break
            # sleep a bit to wait for any result, then repeat.
            time.sleep(1)

            # if you have a progress bar, this is where you move it.
            active_stuff.remove(handle)
            data_out.append(handle.get())
    return data_out

def get_polygon(shapeRec,tolerance):
    numParts = len(shapeRec.shape.parts)
    polys = []
    
    for i in range(numParts):
        ini = shapeRec.shape.parts[i]
        if i+1 < numParts:
            end = shapeRec.shape.parts[i+1]
            polys.append(Polygon(shapeRec.shape.points[ini:end]))
        else:
            polys.append(Polygon(shapeRec.shape.points[ini:]))
    
    # create a multipolygon with all the individual parts
    polygon = MultiPolygon(polys)
    if tolerance != 0:
        polygon = polygon.buffer(args.tolerance)
    return polygon

def processing(pointsfile_text,sf,tolerance):
    
    
    multi = False # are we doing multiprocessing?
    
    pointsfile_lines = pointsfile_text.split("\n")
    pointsfile_lines_length = len(pointsfile_lines)
    print("total",len(pointsfile_lines))
    
    # I think these could be moved inside anyway?
    # hm...
    output_data = []
    outfile_string = ""
    
    # which are shapes describing a region right?
    # iterate over shape-records
    for shapeRec in sf.shapeRecords():
        
        # it's unclear what happens with shapes around the 0
        # meridian. because if the polygon is just defined via points
        # as quasi 2d, this will not work there.
        
        # a shape can be divided in several polygonal parts
        polygon = get_polygon(shapeRec,tolerance)
    
        myenvelope = shapely.envelope(polygon) # calculating envelope first.
        
        if multi:
            data_in = []
            batch_size=100000
            batch_counter = 0
            while batch_counter*batch_size < pointsfile_lines_length-1:
                sub_list = pointsfile_lines[batch_counter * batch_size :(batch_counter+1)*batch_size]
                data_in.append((sublist,myenvelope,polygon))
                batch_counter += 1
            
            data_out = local_multi(data_in)
            
            for my_tuple in data_out:
                sub_string, sub_list = my_tuple
                outfile_string += sub_string
                output_data += sub_list
        
        else:    
            counter = 0

            while counter < pointsfile_lines_length-1:
                # -1 because last line will be empty 
                # because that's how text files work.
            
                pline = pointsfile_lines[counter]
                line, output_sub = single_line_check(pline,myenvelope,polygon)
                outfile_string += line
                
                # this only works if the list is empty if single
                # line check doesn't produce an output.
                output_data += output_sub 
            
                if counter % 100000 == 0:
                    print("pointslining",counter)
                counter += 1
        
        # removed error handling, this should just work, all the time
        # if it fails, we want to know where and why.
            
    print('Found %d peaks'%len(output_data))
    return outfile_string, output_data

def batch_check(lines,myenvelope,polygon):
    sub_string = ""
    sub_list = []
    for line in lines:
        line, output_sub = single_line_check(pline,myenvelope,polygon)
        sub_string += line
        sub_list += output_sub
    return sub_string, sub_list
    
def single_line_check(pline,myenvelope,polygon):
    """
    extracted for possible multiprocessing later
    and testing maybe.
    """
    
    xcol = 1
    ycol = 0
    # not ideal prefer batch conversions.
    vals = pline.split(',')
    point = Point([float(vals[xcol]), float(vals[ycol])])
    
    line = ""
    output_sub = []
    if myenvelope.contains(point):
        if polygon.contains(point):
            line = pline.encode('utf-8', errors='replace').decode('utf-8')
            line += "\n"
            # this needs to be a list with one element with the
            # actual result because of reasons in the loop.
            output_sub = [[float(x) for x in vals]]
            
    return line, output_sub
    
if __name__=="__main__":
    cmd()
