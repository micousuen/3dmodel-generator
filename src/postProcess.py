import os
import math
import argparse
import numpy as np
import scipy.io as io
from functools import reduce
from operator import itemgetter

class ProcessPointCloud():
    def points2ply(self,pointslist):
        write_to_ply = []
        write_to_ply.append("ply")
        write_to_ply.append("format ascii 1.0")
        write_to_ply.append("element vertex "+str(len(point_cloud)))
        write_to_ply.append("property float x")
        write_to_ply.append("property float y")
        write_to_ply.append("property float z")
        write_to_ply.append("end_header")
        write_to_ply.extend([" ".join([str(k) for k in i]) for i in pointslist])
        for i, c in enumerate(write_to_ply):
            if not c.endswith("\n"):
                write_to_ply[i] = c + "\n"
        return write_to_ply

    def readMat2array(self,filepath):
        result = np.array(io.loadmat(filepath)["instance"]).astype(np.int32)
        return result

    def array2points(self, result, process_mod = 1, fill_neighbor_num = 3):
        """
        process_mod:
            1: raw point cloud data
            2: merged labelled point cloud data
            3: merged filtered labelled point cloud data
            4: connect point clouds together based on distance between clouds
        """
        result_shape = result.shape
        point_cloud = []
        connect_component_thickness = 1
        connect_component_extendRange = 3
        # Some help function to check point in volume, to get neighbors
        in_volume = lambda coord: reduce(lambda x, y: x and y, [coord[i] < result_shape[i] and coord[i] >=0 for i in range(len(coord))])
        neighbor_list_x = lambda i, j, k: [p for p in [[i-1, j, k],[i+1,j,k],[i, j-1, k],[i, j+1, k]] if in_volume(p)]
        neighbor_list_y = lambda i, j, k: [p for p in [[i-1, j, k],[i+1,j,k],[i, j, k-1],[i, j, k+1]] if in_volume(p)]
        neighbor_list_z = lambda i, j, k: [p for p in [[i, j, k-1],[i,j,k+1],[i, j-1, k],[i, j+1, k]] if in_volume(p)]
        p_distance = lambda x, y: math.sqrt(reduce(lambda a, b: a+b, [(i-j)**2 for i, j in zip(x, y)]))
        det = lambda x: math.sqrt(reduce(lambda a, b: a+b, [i**2 for i in x])) # To get determinant of a vector
        get_points_between_two_points = lambda x, y: [(x0, x1, x2) \
                                                      for x0 in range(min(x[0], y[0])-connect_component_extendRange, min(x[0], y[0])+connect_component_extendRange+abs(x[0]-y[0])+1) \
                                                      for x1 in range(min(x[1], y[1])-connect_component_extendRange, min(x[1], y[1])+connect_component_extendRange+abs(x[1]-y[1])+1) \
                                                      for x2 in range(min(x[2], y[2])-connect_component_extendRange, min(x[2], y[2])+connect_component_extendRange+abs(x[2]-y[2])+1)\
                                                      if det(( (x1-x[1])*(x2-y[2])-(x2-x[2])*(x1-y[1]), \
                                                             ( (x2-x[2])*(x0-y[0])-(x0-x[0])*(x2-y[2]) ), \
                                                             ( (x0-x[0])*(x1-y[1])-(x1-x[1])*(x0-y[0]) )))\
                                                        /det((x[0]-y[0], x[1]-y[1], x[2]-y[2]))<math.sqrt(3)*connect_component_thickness\
                                                      and (x0 != x[0] or x1 != x[1] or x2 != x[2]) \
                                                      and (x0 != y[0] or x1 != x[1] or x2 != x[2]) \
                                                      and in_volume((x0, x1, x2))] # points between two points by calculating the distance to the line

        # If more than 3 block exist in one plane, then make fulfill the hole
        def fill_hole(dataArray, neighbor_num = 3):
            hole_filled = 0
            for i in range(result_shape[0]):
                for j in range(result_shape[1]):
                    for k in range(result_shape[2]):
                        if reduce(lambda x, y: x+y, [1 if dataArray[p[0], p[1], p[2]]!=0 else 0 for p in neighbor_list_x(i, j, k)]) >=neighbor_num and dataArray[i, j, k] == 0:
                            dataArray[i, j, k] = 1
                            hole_filled += 1
                        if reduce(lambda x, y: x+y, [1 if dataArray[p[0], p[1], p[2]]!=0 else 0 for p in neighbor_list_y(i, j, k)]) >=neighbor_num and dataArray[i, j, k] == 0:
                            dataArray[i, j, k] = 1
                            hole_filled += 1
                        if reduce(lambda x, y: x+y, [1 if dataArray[p[0], p[1], p[2]]!=0 else 0 for p in neighbor_list_z(i, j, k)]) >=neighbor_num and dataArray[i, j, k] == 0:
                            dataArray[i, j, k] = 1
                            hole_filled += 1
            return (dataArray, hole_filled)
        def iterative_fill_hole(dataArray, neighbor_num = 3):
            # iterative fill holes
            iteration_limit = 1 if neighbor_num < 3 else 10
            for i in range(iteration_limit): # at most 10 iterations, can exit early
                dataArray, filled_num = fill_hole(dataArray, neighbor_num)
                print("At fill hole iteration ", i, " , filled ", filled_num, " holes")
                if filled_num == 0:
                    break
            return dataArray
        def label_model(dataArray):
            label_data = []
            label_num = 0
            dataArray[dataArray>0] = -1
            for i in range(result_shape[0]):
                for j in range(result_shape[1]):
                    for k in range(result_shape[2]):
                        if dataArray[i, j, k] >= 0:
                            continue
                        else:
                            # -1. Point unlabelled
                            label_data.append([])
                            neighbor_stack = [(i, j, k)]
                            for p in neighbor_stack:
                                def check_add_point(a, b, c):
                                    if in_volume((a, b, c)) and dataArray[a, b, c] == -1 and not [a, b, c] in neighbor_stack:
                                        neighbor_stack.append(([a, b, c]))
                                label_data[label_num].append(p)
                                dataArray[p[0], p[1], p[2]] = label_num+1
                                check_add_point(p[0]-1, p[1], p[2])
                                check_add_point(p[0]+1, p[1], p[2])
                                check_add_point(p[0], p[1]-1, p[2])
                                check_add_point(p[0], p[1]+1, p[2])
                                check_add_point(p[0], p[1], p[2]-1)
                                check_add_point(p[0], p[1], p[2]+1)
                            label_num += 1
            point_cloud = [item for l in label_data for item in l]
            return (point_cloud, label_data)
        def filter_point_cloud(label_data, filter_size = 10):
            filtered_labelled_cloud = list(filter(lambda x: len(x)>filter_size, label_data))
            point_cloud = [item for l in filtered_labelled_cloud for item in l]
            return (point_cloud, filtered_labelled_cloud)
        def connect_disconnected(label_data):
            cloud_min_dis_points = [None for _ in label_data]
            for i, label_group in enumerate(label_data):
                # for each labelled group, find out the cloest point in other groups, use reduce to get cloest point for this group
                otherpoints = [(p, i0, i1) for i0, g in enumerate(label_data) if i0 != i for i1, p in enumerate(g)]
                currentpoints = [p for p in label_group]
                currentpoints_dis = [None for _ in currentpoints]
                for j, p1 in enumerate(currentpoints):
                    # for each point in this group, calculate the cloest point in other groups, use reduce to get cloest point for each point
                    distance_list = [(p_distance(p1, p2[0]), p1, p2[0], p2[1], p2[2]) for p2 in otherpoints]
                    min_dis = reduce(lambda x, y: x if x[0] < y[0] else y, distance_list)
                    currentpoints_dis[j] = min_dis
                cloud_min_dis = reduce(lambda x, y: x if x[0] < y[0] else y, currentpoints_dis)
                cloud_min_dis_points[i] = cloud_min_dis
            # cloud_min_dis_points will store cloest point for each group, use cloud_weight to get which group have highest weight in connecting others
            cloud_weight = [[0, i] for i in range(len(label_data))]
            for c_i in cloud_min_dis_points:
                cloud_weight[c_i[3]][0] += 1
            cloud_weight = sorted(cloud_weight, key=itemgetter(0), reverse = True)
            # Calculate distance to the line between one pair of cloest points and add points to connect them
            points_added = []
            for p_pair in cloud_min_dis_points:
                points_added.extend(get_points_between_two_points(p_pair[1], p_pair[2]))
            return points_added
        def points2dataArray(points):
            dataArray = np.zeros(result_shape)
            for p in points:
                if in_volume(p):
                    dataArray[p[0], p[1], p[2]] = 1
            return dataArray
        def dataArray2points(dataArray):
            points = []
            for i in range(result_shape[0]):
                for j in range(result_shape[1]):
                    for k in range(result_shape[2]):
                        if dataArray[i, j, k] > 0 and in_volume((i, j, k)):
                            points.append((i, j, k))
            return points

        if process_mod == 1:
            point_cloud = dataArray2points(result)
        elif process_mod == 2:
            point_cloud, label_data = label_model(result)
        elif process_mod == 3:
            point_cloud, label_data = label_model(result)
            point_cloud, filtered_labelled_cloud = filter_point_cloud(label_data, filter_size = 15)
        elif process_mod == 4: 
            point_cloud, label_data = label_model(result)
            point_cloud, filtered_labelled_cloud = filter_point_cloud(label_data, filter_size = 15)
            count = 0
            while len(filtered_labelled_cloud) > 1 or count > 5:
                print("At connect disconnected iteration ", count, " , there are ", len(filtered_labelled_cloud), " clusters")
                points_to_add = connect_disconnected(filtered_labelled_cloud)
                dataArray = points2dataArray([p for g in filtered_labelled_cloud for p in g]+points_to_add)
                point_cloud, filtered_labelled_cloud = label_model(dataArray)
                count += 1
        else:
            raise ValueError("Unknown process_mod")

        point_cloud = dataArray2points(iterative_fill_hole(points2dataArray(point_cloud), neighbor_num=fill_neighbor_num))
        return point_cloud

    def VF_compress(self, obj_v, obj_f, compress_level = 1):
        '''
        Level 1: No compress
        Level 2: Compress vertices
        Level 3: Compress vertices, Compress faces
        '''
        if compress_level == 1:
            return obj_v, obj_f
        elif compress_level == 2:
            point_set = list(set(obj_v))
            point_dict = {c:i for i, c in enumerate(point_set, 1)}
            new_f = [None for _ in range(len(obj_f))]
            for i, c in enumerate(obj_f):
                new_list = []
                for v in c:
                    new_list.append(point_dict[obj_v[v-1]])
                new_f[i] = tuple(sorted(new_list))
            return point_set, new_f
        elif compress_level == 3:
            point_set = list(set(obj_v))
            point_dict = {c:i for i, c in enumerate(point_set, 1)}
            new_f = [None for _ in range(len(obj_f))]
            for i, c in enumerate(obj_f):
                new_list = []
                for v in c:
                    new_list.append(point_dict[obj_v[v-1]])
                new_f[i] = tuple(sorted(new_list))
            new_f = list(set(new_f))
            return point_set, new_f
        else:
            return obj_v, obj_f

    def points2VF(self,point_list, cube_len):
        point_cloud = [[cube_len*i for i in p] for p in point_list]
        result = []
        result_v = []
        result_f = []
        base_num = 0
        for p in point_cloud:
            result_v.append((p[0]+cube_len*0, p[1]+cube_len*0, p[2]+cube_len*0))
            result_v.append((p[0]+cube_len*0, p[1]+cube_len*0, p[2]+cube_len*1))
            result_v.append((p[0]+cube_len*0, p[1]+cube_len*1, p[2]+cube_len*0))
            result_v.append((p[0]+cube_len*0, p[1]+cube_len*1, p[2]+cube_len*1))
            result_v.append((p[0]+cube_len*1, p[1]+cube_len*0, p[2]+cube_len*0))
            result_v.append((p[0]+cube_len*1, p[1]+cube_len*0, p[2]+cube_len*1))
            result_v.append((p[0]+cube_len*1, p[1]+cube_len*1, p[2]+cube_len*0))
            result_v.append((p[0]+cube_len*1, p[1]+cube_len*1, p[2]+cube_len*1))
            result_f.append((base_num+1, base_num+2, base_num+4))
            result_f.append((base_num+1, base_num+3, base_num+4))
            result_f.append((base_num+1, base_num+2, base_num+6))
            result_f.append((base_num+1, base_num+5, base_num+6))
            result_f.append((base_num+1, base_num+3, base_num+5))
            result_f.append((base_num+3, base_num+5, base_num+7))
            result_f.append((base_num+2, base_num+4, base_num+6))
            result_f.append((base_num+4, base_num+6, base_num+8))
            result_f.append((base_num+5, base_num+6, base_num+8))
            result_f.append((base_num+5, base_num+7, base_num+8))
            result_f.append((base_num+3, base_num+4, base_num+8))
            result_f.append((base_num+3, base_num+7, base_num+8))
            base_num += 8
        return result_v, result_f 

    def VF2objFormat(self, result_v, result_f):
        result = []
        result.extend([" ".join(["v"]+[str(i) for i in p]) for p in result_v])
        result.extend([" ".join(["f"]+[str(i) for i in f]) for f in result_f])
        for i, c in enumerate(result):
            if not c.endswith('\n'):
                result[i] = c+"\n"
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="post processing module to tranform generate mat file to obj file")
    parser.add_argument("-p", "--path", default="./voxel.png.mat", dest="rootpath", help="mat file path")
    parser.add_argument("-n", "--number", default=3, type=int, dest="neighbor_num", help="Define number of neighbors in filling holes")
    args = parser.parse_args()

    p = ProcessPointCloud()
    file_name = args.rootpath
    point_cloud = p.array2points(p.readMat2array(file_name), 4, fill_neighbor_num=args.neighbor_num)
    result = p.VF2objFormat(*p.VF_compress(*p.points2VF(point_cloud, 0.1), 3))
    with open(os.path.splitext(file_name)[0]+".obj", "w") as f:
        f.writelines(result)