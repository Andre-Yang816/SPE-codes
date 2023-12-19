# coding=utf-8

import os


class configuration_file():
    def __init__(self):
        self.rootpath = os.path.dirname(os.getcwd())
        self.datafolderPath = os.path.join(self.rootpath, "Data")    #The folder where the original csv data is stored
        self.saveResultsPath = os.path.join(self.rootpath, 'test_result')
        self.jar_path = os.path.join(self.rootpath, "RankLib.jar")
        self.weka_jar_path = os.path.join(self.rootpath, "myregression.jar")
        # The total folder that stores arff files, which contains several subfolders,
        # including classification and regression arff files.
        self.arffDateFolder = os.path.join(self.rootpath, "ArffData")

        self.crossversiondatafolderPath = os.path.join(self.rootpath, "CrossversionData")
        # When crossverion training is used, for the folder where the original csv data is stored, refer to
        # the annotation of import_crossversion_data under the Processing class.

        self.bootstrap_count = 1       # Number of bootstraps
        self.crossverion_count = 1     # Number of cross-version data runs

        self.bootstrap_dir = os.path.join(self.rootpath, "crossrelease_csv")     # bootstrap csv storage folder
        self.is_remain_origin_bootstrap_csv = True
        # Do you want to retain the results of previous bootstrap?
        # False: Not retained True: Reserved

        # Folder to save the number indicator results
        self.save_PredBugCountResult_dir = os.path.join(self.rootpath, "PredBugCountResult")
        # Folder to save density indicator results
        self.save_PredBugDensityResult_dir = os.path.join(self.rootpath, "PredBugDensityResult")
        self.save_PredBugCCResult_dir = os.path.join(self.rootpath, "PredBugCCResult")

        pass

    def getrootpath(self, a):
        return self.rootpath
