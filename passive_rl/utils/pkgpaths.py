
import os
import passive_rl

 
class PkgPath(object):

    PACKAGE_PATH = passive_rl.__path__[0] 

    def trainingdata(file=""):
        return os.path.join(PkgPath.PACKAGE_PATH, os.pardir, "data/training", file) 

    def modelsxml(file=""):
        return os.path.join(PkgPath.PACKAGE_PATH, os.pardir,  "data/xml", file) 

  