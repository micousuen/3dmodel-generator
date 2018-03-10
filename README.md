# 3D-printable model generator

* dataIO.py: data input and output model, and can transform mesh model to voxel model
* setting.py: global setting variables
* view.py: tools to visualize model or result
* utils.py: some auxiliary functions, all other class will inherit this class

## Package Requirement
1. pathos: for multiprocessing, get from "pip install git+https://github.com/uqfoundation/dill.git@master" and "pip install git+https://github.com/uqfoundation/pathos.git@master"
2. trimesh: to read model and transform it, get from "pip install trimesh"

## ChangeLog
20180310
* Fix some training problems. 

20180309
* Finished training procesure. First runable model

20180308
* Finished model construction. 

20180307
* Finished view module. Can use mat files to generate model images

20180306 
* Finished dataIO module, can read and write model, can transform mesh model to voxel model and save them. Can randomly yield models. 

