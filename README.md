# Bug Numbers Matter: An Empirical Study of Effort-Aware Defect Prediction Using Class Labels vs. Bug Numbers

## code
If you want to test directly based on the model we provide, execute:
```
$ python testModel_count_CrossProject.py
$ python testModel_count_CrossRelease.py
$ python testModel_Density_CrossProject.py
$ python testModel_Density_CrossRelease.py
```
Among them, testModel_count_CrossProject.py refers to the test module as the effort, and the effort-aware indicator value of each algorithm under the Cross-Project setting.
In the same way, testModel_count_CrossRelease.py, the test uses module as the effort, and the effort-aware indicator value of each algorithm under the Cross-Release setting
testModel_Density_CrossProject.py, testModel_Density_CrossRelease.py respectively test the effort-aware indicator value of each algorithm using LOC as the effort under the Cross-Project and Cross-Release settings respectively.

If you need to train by yourselves, execute:

```
$ python trainModel_count_CrossProject.py
$ python trainModel_count_CrossRelease.py
$ python trainModel_Density_CrossProject.py
$ python trainModel_Density_CrossRelease.py
```
In this way, the models under each setting can be obtained, and finally 120 models will be obtained (2 effort × 2 settings × 3 algorithms × 10 times).
This process is time consuming, but don’t worry, we provide all trained models in the 'Model' directory.

## data
We provide 41 PROMISE datasets
Under Cross-Release settings, simple processing is required,
Execute the following code to complete the division of cross-release train-test data set pairs
```
$ python devideDataSet.py
```
Under the cross-project setting, you can directly extract the original version of each data set in the Data directory as a data set. A total of 11 data sets can be obtained. The specific operations are mentioned in the paper.

## model
We provide all models when LOC is used as effort, including all models under two settings: cross-release and cross-project settings. You can directly unzip the compressed package in the model directory to obtain it.
Due to space limitations when uploading, the module cannot be successfully uploaded as an effort model. We are looking for another cloud disk to save the model. If you have any needs, please contact me through the following email: 284425@whut.edu.cn
