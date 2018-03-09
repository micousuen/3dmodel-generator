'''
Created on Mar 5, 2018

@author: micou
'''
import os
import argparse
import numpy as np
import scipy.io as io
from functools import reduce
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except:
    print("Warning: Require pathos package, if package missing, cannot generate Voxelmodels using multiprocessing\n\
                    pip install git+https://github.com/uqfoundation/dill.git@master, \n\
                    pip install git+https://github.com/uqfoundation/pathos.git@master")
try:
    import trimesh
except:
    print("Fatal: Require trimesh package, check https://pypi.python.org/pypi/trimesh")
    exit()


from utils import Utils
from setting import SHAPENET_MODEL_ROOTPATH

class DataIO(Utils):
    rootpath = "./"
    cate_dir = []
    cate_info = {} # {<category>:{informations}}
    model_dir = {} # {<category>:[<model_foldername>, <model_foldername>...]}
    default_modelInfo_filename = "model_normalized.json"
    default_model_filename = "model_normalized.obj"
    
    # Processing information
    processed = 0
    total_modelnum = 0
    
    def __init__(self, rootpath=rootpath, certain_cate=[], saved_model_dirfile="./model_dir.json"):
        """
        Strongly recommend to check rootpath before running. Otherwise you will get bunch of warnings
        """
        self.rootpath = os.path.abspath(rootpath)
        self._check_validation()
        self._read_model_dir(certain_cate, saved_model_dirfile)
        #pprint.pprint(self.model_dir)
        
    def _check_validation(self):
        if not os.path.isdir(self.rootpath):
            self.error("Root path do not exist")
    
    def _read_model_dir(self, certain_cate=[], saved_model_dirfile="./model_dir.json"):
        def _build_model_dir(certain_cate):
            self.cate_dir = [os.path.join(self.rootpath, n) \
                             for n in os.listdir(self.rootpath) \
                             if os.path.isdir(os.path.join(self.rootpath, n)) and (n in certain_cate or certain_cate==[] ) ]
            for cate in self.cate_dir:
                # if there are model folder under each model directory, that means this shapeNet raw database.
                # Otherwise this is generated mat database
                paths = os.listdir(cate)
                temp_model = []
                for p in paths:
                    if os.path.isdir(os.path.join(cate, p, "models")):
                        # If mesh model
                        temp_model.append(os.path.join(cate, p, "models"))
                    elif p.endswith(".mat"):
                        # If already mat file
                        temp_model.append(os.path.join(cate, p))
                self.model_dir[cate] = sorted(temp_model)
                
        if os.path.isfile(saved_model_dirfile):
            try:
                store_info = self.read_from_json(saved_model_dirfile)
                self.cate_dir = store_info["cate_dir"]
                self.model_dir = store_info["model_dir"]
            except:
                _build_model_dir(certain_cate)
                self.write_to_json({"cate_dir":self.cate_dir, "model_dir":self.model_dir}, saved_model_dirfile)
        else:
            _build_model_dir(certain_cate)
            self.write_to_json({"cate_dir":self.cate_dir, "model_dir":self.model_dir}, saved_model_dirfile)
            
    def get_cate(self):
        """
        return a copy of all categories
        """
        return [n for n in self.cate_dir]
    
    def get_numOfModels(self, cate=""):
        """
        return number of models of a certain cate, if not given, return number of all models
        """
        if cate=="":
            return reduce(lambda n, key: n+len(self.model_dir[key]), self.model_dir, 0)
        elif cate in self.model_dir:
            return len(self.model_dir[cate])
        else:
            return 0
    
    def get_cateListFromCates(self, cates=""):
        """
        return a category list from cates
        if no cates given, will return all cates
        if give multi cates, cates should be seperated by space
        if there is not such a cate, return []
        """
        temp_cate_list = [m for n in cates.split() if reduce(lambda b, l:b or (n in l), self.get_cate(), False) \
                            for m in self.get_cate() if n in m]
        if cates == "":
            cate_list = self.get_cate()
        elif temp_cate_list != []:
            cate_list = temp_cate_list
            if len(cate_list) != len(cates.split()):
                self.warn("Some category not exist, check your input")
        else:
            cate_list = []
        return cate_list
    
    def get_flattenAbsModelDir(self, cates=""):
        """
        return a list of model dirs in cates, if cates not given return all models' dir
        """
        cate_list = self.get_cateListFromCates(cates)
        model_dir_list = reduce(lambda l, key: l+[m for m in self.model_dir[key]], cate_list, [])
        return model_dir_list

    def _trans_shiftPadding(self, inputNdArray, offset, shape, padding_value = 0):
        """
        Copy inputNdArray to new space, and it will shift on new space based on offset given.
        All other space of that new space will be padded by padding_value
        return a new ndarray with shape, and it's type is the same as inputNdArray
        """
        if len(offset) != len(shape) or len(offset) != len(inputNdArray.shape):
            self.error("Shift Padding input size not match")
        result = np.full(shape, padding_value)
        result.astype(inputNdArray.dtype)
        result_slices = [slice(offset[dim], offset[dim]+inputNdArray.shape[dim]) for dim in range(inputNdArray.ndim)]
        result[result_slices] = inputNdArray
        return result
    
    def transfrom_meshmodel2voxel(self, meshmodel, dim=64):
        """
        transfer a meshmodel(in trimesh type) to voxelized model
        """
        # Return Boolean ndarray, get voxelized model, can use other methods. Input meshmodel should be normalized
        voxel = meshmodel.voxelized(pitch=1/(dim-2)).matrix 
        
        # Copy voxel model to (dim, dim, dim) space and centralize it
        shift_direction = ((np.array((dim, dim, dim))-np.array(voxel.shape))/2).astype(np.int)
        shift_voxel = self._trans_shiftPadding(voxel, shift_direction, (dim, dim, dim), 0)
        return shift_voxel
    
    def transform_saveVoxelFile(self, meshmodel_filepath, dim, dest_samedir, dest_filename, dest_dir=""):
        """
        transform a meshmodel to voxelmodel. Process one model at one time
        """
        meshmodel_dir, _ = os.path.split(meshmodel_filepath)
        info_filepath = os.path.join(meshmodel_dir, "model_normalized.json")
        if dest_samedir:
            dest_directory, _ = os.path.split(meshmodel_filepath)
            output_dir = os.path.join(dest_directory, dest_filename)
        else:
            mo, _ = os.path.split(meshmodel_filepath) # ca_dir/models, _
            mo_dir, _ = os.path.split(mo) # model_dir, models
            ca_dir, mo_no = os.path.split(mo_dir) # cate_dir, model_no
            _, cate = os.path.split(ca_dir) # rootpath, cate
            if not os.path.isdir(os.path.join(dest_dir, cate)):
                os.mkdir(os.path.join(dest_dir, cate))
            output_dir = os.path.join(dest_dir, cate, mo_no+"_"+dest_filename)
        #Model file only
        if os.path.isfile(output_dir):
            self.info("Already Exist "+output_dir)
        else:
            modelinfo = self.get_modelInfo(info_filepath)
            self.info("Vertices <"+str(modelinfo["numVertices"])+"> "+meshmodel_filepath+"\n-->"+output_dir)
            voxel_model = self.transfrom_meshmodel2voxel(self.get_model(meshmodel_filepath), dim)
            io.savemat(output_dir, {"instance":voxel_model, "info":modelinfo}, appendmat=True, do_compression=True)
        
    def transform_saveVoxelFiles(self, cates="", source_filename = "model_normalized.obj", \
                             dest_filename="model_normalized.mat", dim=64, multiprocess=4, dest_samedir=True, dest_dir=""):
        """
        Use map function to generate voxel models, you may only need this once and this will take a long time for transformation
        if dest_samedir is False, then dest_dir should be given
        """
        if not dest_samedir:
            if dest_dir == "":
                self.warn("Destination directory not given, use default dest_dir which will under current dir")
                dest_dir = "./voxelModels"
            dest_dir = os.path.abspath(dest_dir)
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
        
        # Use multi-processor to transform models. 
        # will only accept meshmodel because we will check source_file existence 
        # Only when there are obj file this can work
        model_paths = [os.path.join(p, source_filename) for p in self.get_flattenAbsModelDir(cates) \
                       if os.path.isfile(os.path.join(p, source_filename)) and source_filename.endswith(".obj") ]
        self.info("Done model path building")
        
        if multiprocess > 1:
            # If package not given, will not be able to use this multiprocessing
            ProcessPool = Pool(multiprocess)
            ProcessPool.map(lambda x: self.transform_saveVoxelFile(x, dim, dest_samedir, dest_filename, dest_dir), model_paths)
            ProcessPool.close()
            ProcessPool.join()
            ProcessPool.terminate()
        else:
            # Use only one thread to process mesh model to voxel model
            for c, path in enumerate(self.random_permutation(model_paths)):
                self.transform_saveVoxelFile(path, dim, dest_samedir, dest_filename, dest_dir)
                self.info("Process: {0}/{1}".format(c, len(model_paths)))
    
    def get_model_abspath(self, model_dir_info, filename):
        if isinstance(model_dir_info, tuple):
            (cate, model_num)=model_dir_info
            if self.model_dir[cate][model_num].endswith(".mat") or self.model_dir[cate][model_num].endswith(".json"):
                return self.model_dir[cate][model_num]
            else:
                return os.path.join(self.model_dir[cate][model_num], filename)
        elif isinstance(model_dir_info, str):
            if model_dir_info.endswith(".mat") or model_dir_info.endswith(".json"):
                return model_dir_info
            else:
                return os.path.join(model_dir_info, filename)
        else:
            self.error("Error model_dir_info")
    
    def get_model(self, filepath):
        """
        return model based on file type
        """
        if not os.path.isfile(filepath):
            self.warn(filepath+" not exist")
            return None
        _, fileExt = os.path.splitext(filepath)
        if fileExt == ".obj":
            return trimesh.load(filepath)
        elif fileExt == ".json":
            return self.read_from_json(filepath)
        elif fileExt == ".mat":
            return io.loadmat(filepath)["instance"]
        else:
            self.error("Unknown model type")
    
    def get_modelInfo(self, filepath):
        """
        return model info, the lowest level function
        """
        if not os.path.isfile(filepath):
            self.warn(filepath+" not exist")
            return None
        else:
            return self.read_from_json(filepath)
    
    def get_model_byModelNum(self, cate, model_num, filename=default_model_filename):
        """
        return mesh model
        filename only valid when target model_dir is not endswith .mat
        """
        filepath = os.path.join(self.get_model_abspath((cate, model_num), filename))
        return self.get_model(filepath)
    
    def get_modelInfo_byModelNum(self, cate, model_num, filename=default_modelInfo_filename):
        """
        return model information
        filename only valid when target model_dir is not endswith .mat
        """
        filepath = os.path.join(self.get_model_abspath((cate, model_num), filename))
        return self.get_modelInfo(filepath)
    
    def get_model_byModelName(self, cate, model_name, filename=default_model_filename):
        """
        return mesh model, alias of get_model
        filename only valid when target model_dir is not endswith .mat
        """
        return  self.get_model_byModelNum(cate, self.model_dir[cate].index(model_name), filename)
    
    def get_modelInfo_byModelName(self, cate, model_name, filename=default_modelInfo_filename):
        """
        return model information, alias of get_model
        filename only valid when target model_dir is not endswith .mat
        """
        return self.get_modelInfo_byModelNum(cate, self.model_dir[cate].index(model_name), filename)
    
    def get_voxmodel_generator(self, cates="", random=False):
        """
        return a generator which will return (voxel model, {}) with model info missing
        if random, then randomly select model, but everyone will be selected in the end
        if cates not given, return all models we have. To input multi category, use space to seperate them
        """
        # Generate category list to iterate
        cate_list = self.get_cateListFromCates(cates)
        if cate_list == []:
            raise StopIteration
        
        if random:
            # Randomly permute category, and randomly permute every model
            model_list = self.get_flattenAbsModelDir(cates)
            for md in self.random_permutation(model_list):
                if not md.endswith(".mat"):
                    self.warn("voxel model error at path: "+md)
                else:
                    yield (self.get_model(md), {"filepath":md})
        else:
            model_list = self.get_flattenAbsModelDir(cates)
            for md in model_list:
                if not md.endswith(".mat"):
                    self.warn("voxel model error at path: "+md)
                else:
                    yield (self.get_model(md),{"filepath":md} )
    
    def get_meshmodel_generator(self, cates="", random=False, obj_filename=default_model_filename, info_filename=default_modelInfo_filename):
        """
        return a generator which will return (mesh model, model info)
        if random, then randomly select model, but everyone will be selected in the end
        if cates not given, return all models we have. To input multi category, use space to seperate them
        
        """
        # Generate category list to iterate
        cate_list = self.get_cateListFromCates(cates)
        if cate_list == []:
            raise StopIteration
        
        if random:
            # Randomly permute category, and randomly permute every model
            model_list = self.get_flattenAbsModelDir(cates)
            for md in self.random_permutation(model_list):
                if md.endswith(".mat") or md.endswith(".json"):
                    self.warn("mesh model path error: "+md)
                else:
                    yield (self.get_model(self.get_model_abspath(md, obj_filename)), self.get_modelInfo(self.get_model_abspath(md, info_filename)))
        else:
            model_list = self.get_flattenAbsModelDir(cates)
            for md in model_list:
                if md.endswith(".mat") or md.endswith(".json"):
                    self.warn("mesh model dir error: "+md)
                else:
                    yield (self.get_model(self.get_model_abspath(md, obj_filename)), self.get_modelInfo(self.get_model_abspath(md, info_filename)))
                    
    def get_batchmodels(self, cates="", batchNum=100, epoch=100, random=False, modelType="voxel", obj_filename=default_model_filename, info_filename=default_modelInfo_filename):
        """
        return a batch of models (with no info). Use get_voxelmodel_generator and get_meshmodel_generator to get model
        Go through whole data set at most <epoch> times
        modelType is string in {"voxel","mesh"}
        obj_filename and info_filename are optional, only valid when modelType is 'mesh'
        """
        if modelType == "voxel":
            generator = lambda : self.get_voxmodel_generator(cates, random)
        elif modelType == "mesh":
            generator = lambda : self.get_meshmodel_generator(cates, random, obj_filename, info_filename)
        else:
            self.error("Unknown modelType in get_batchmodels")
            return
        
        epoch_count = 0
        result_list = []
        while epoch_count < epoch:
            modelpool = generator()
            for modelAndInfo in modelpool:
                if len(result_list) < batchNum:
                    result_list.append(modelAndInfo[0])
                else:
                    yield result_list[:batchNum]
                    result_list = []
            epoch_count += 1
        if len(result_list) > 0:
            yield result_list
        
        
if __name__== "__main__":
#     SHAPENET_MODEL_ROOTPATH = "./test_folder"
#     SHAPENET_MODEL_ROOTPATH = "./test_mat"
    parser = argparse.ArgumentParser(description="dataIO module to tranform mesh model to voxel model")
    parser.add_argument("-p", "--path", default=SHAPENET_MODEL_ROOTPATH, dest="rootpath", help="Root path to unzipped shapeNet folder")
    parser.add_argument("-n", "--number", default=1, type=int, dest="processors_number", help="Define number of processors to do transform")
    parser.add_argument("-m", "--modeltree", default="./model_dir.json", dest="model_dir_filename", help="json file that saved model directory, if not exist, create one after model tree building")
    args = parser.parse_args()
    
    Utils().info("Program start")
    test = DataIO(args.rootpath, [], args.model_dir_filename )
    
#     for m in test.get_batchmodels("", 100, 5, True, "voxel"):
#         print(len(m))
        
    test.transform_saveVoxelFiles("", dim=64, multiprocess=args.processors_number, dest_samedir=False, dest_dir="./voxelModels")

        
        
