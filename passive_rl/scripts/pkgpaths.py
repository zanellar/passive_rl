
import os
import passive_rl
 
class PkgPath():

    _PACKAGE_PATH = passive_rl.__path__[0]  

    ENV_DESC_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/envdata")  
    OUT_TRAIN_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/training")  
    OUT_TEST_FOLDER = os.path.join(_PACKAGE_PATH, os.pardir, "data/testing")  
  