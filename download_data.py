from roboflow import Roboflow
rf = Roboflow(api_key="7jOxVge08fORkeeOnk8D")
project = rf.workspace("mohamed-gcajq").project("safety-helmet-q3b8o-eilxb")
version = project.version(2)
dataset = version.download("yolov11")