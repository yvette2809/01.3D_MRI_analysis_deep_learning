# 3D_MRI_analysis_deep_learning


**Description:** 
About 10,000 brain structure MRI and their clinical phenotype data is available. Some MRI are longitudinal (each participant was followed up several times). MRI data has been preprocessed using standard brain imaging analysis pipeline (denoised, bias corrected, and spatially warped into the standard space).


**Goal:**
- *Presymptomatic prediction*: Classifying AD from healthy individuals is easy. Even non-deep learning methods can easily give more than  90% of accuracy. Also, this classification has minimal clinical utility (too late). What is clinically useful is to make a prediction of (future) risk for AD when an individual shows minimal to no symptoms of cognitive impairment. This prediction will have clinical impact because it will enable early intervention to delay the course of disease. 
- *Interpretable model*: A black box model has little clinical utility to a clinician. A useful model is the one providing information about underlying brain features resulting in the decision (made by a model).


**Datasets:** 
- NACC (National Alzheimer Coordinating Center) has ~8000 MRI sessions each of which may have multiple runs of MRI. Some patients have longitudinal follow-ups. Patients and healthy controls. Clinical data (label data) is available. 
- OASIS (Open Access Series of Imaging Studies) has ~2000 MRI. Patients and healthy controls. Clinical data (label data) is available.


**Model selection:**
- 3D Convolutional Autoencoder: use this if needed.
- 3D Convolutional Neural Networks: the primary model with ReLU activation and Xavier initialization of filter parameter for each convolutional layer, max pooling method for the pooling layer, and softmax for the flattened layer.


**Accomplishments:**
- Computing resource setup:
  1. Migrated to supercomputer environment, successfully accessed stampede2 via jupyter notebook using Python 3 and installed all required packages;
  2. Copied nacc data sets to our own work directory in the supercomputer for further use as recommended by Prof. Cha;
  3. Created a copy of data in scratch library to get faster computation.
- Model building:
  1. Applied the 3D convolutional layers to build a 3D Convolutional Autoencoder, still fixing bugs;
  2. Built a 3D Convolutional Neural Network and applied it on a sample of 3 on our local machine.


**To-do list:**
- Scaling up on the supercomputer:
  1. Install all relevant packages besides tensorflow;
  2. Set up multiple computing nodes on TACC Stampede2 supercomputer.
- Improvement on the model itself:
  1. More detailed data preprocessing: try to eliminate the skull of each MRI image before performing our model;
  2. Add more demographic features: e.g. age, sex, and other demographic data available;
  3. Model adjustments:
      - Add more layers or number of neurons on each layer if training error is high;
      - Add dropouts if the model overfits the training data;
      - Apply cross-validation if needed.
  4. Do research on LRP and consider if it should be used in this project.
