# 3D_MRI_analysis_deep_learning


**Description:** 
About 10,000 brain structure MRI and their clinical phenotype data is available. Some MRI are longitudinal (each participant was followed up several times). MRI data has been preprocessed using standard brain imaging analysis pipeline (denoised, bias corrected, and spatially warped into the standard space).

** **

**Goal:**
- *Presymptomatic prediction*: Classifying AD from healthy individuals is easy. Even non-deep learning methods can easily give more than  90% of accuracy. Also, this classification has minimal clinical utility (too late). What is clinically useful is to make a prediction of (future) risk for AD when an individual shows minimal to no symptoms of cognitive impairment. This prediction will have clinical impact because it will enable early intervention to delay the course of disease. 
- *Interpretable model*: A black box model has little clinical utility to a clinician. A useful model is the one providing information about underlying brain features resulting in the decision (made by a model).

** **

**Datasets:** 
- NACC (National Alzheimer Coordinating Center) has ~8000 MRI sessions each of which may have multiple runs of MRI. Some patients have longitudinal follow-ups. Patients and healthy controls. Clinical data (label data) is available. 
- OASIS (Open Access Series of Imaging Studies) has ~2000 MRI. Patients and healthy controls. Clinical data (label data) is available.

** **

**Model selection:**
- 3D Convolutional Autoencoder;
- 3D Convolutional Neural Networks: the primary model with ReLU activation and Xavier initialization of filter parameter for each convolutional layer, max pooling method for the pooling layer, and softmax for the flattened layer.

** **

**Accomplishments:**
- Computing resource setup:
  1. Migrated to supercomputer environment, successfully accessed stampede2 via jupyter notebook using Python 3 and installed all required packages;
  2. Copied nacc data sets to our own work directory in the supercomputer for further use as recommended by Prof. Cha;
  3. Created a copy of data in scratch library to get faster computation.
- Model building:
  1. Applied the 3D convolutional layers to build a 3D Convolutional Autoencoder, still fixing bugs;
  2. Built a 3D Convolutional Neural Network and applied it on a sample of 3 on our local machine;
- Model modification (on a larger scale of data):
  1. Configured nodes and cores per node needed on supercomputer stampede2;
  2. Applied the model on a data set of 30 images, which is 6 images for each class, and splited the training and test set randomly;
  3. Used mini-batch method with a batch size of 5, and ran 5 epochs to track the change of the cost.

** **

**Contents of the repo**
- CAEs:
  1. CAE_googlecloud.py: the CAE model we used to do test runs on Google Cloud
  2. CAE_stampede2.py: the CAE model we used to run on Stampede2
- CNNs:
  1. 3classes_CNN_googlecloud.py: the 3-class CNN model we used to do test runs on Google Cloud
  2. 3classes_CNN_stampede2.py: the 3-class CNN model we used to run on Stampede2
  3. 5classes_CNN_stampede2.py: the 5-class CNN model we used to run on Stampede2
  4. deepCNN.py: a very deep CNN model with 2 fully connected layers and 21 layers in total
- descriptive data analysis: codes to do descriptive analysis on the NACC dataset
- scratch: codes generated during the whole project process
- supercomputer setup and file transfer:
  1. File Transfer within stampede2.ipynb
  2. Initial Trials and Errors - 1.ipynb
  3. Multi Node Test via Jupyter- Fail, No Permission.ipynb
