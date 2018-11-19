# 3D_MRI_analysis_deep_learning


**Description:** 

About 10,000 brain structure MRI and their clinical phenotype data is available. Some MRI are longitudinal (each participant was followed up several times). MRI data has been preprocessed using standard brain imaging analysis pipeline (denoised, bias corrected, and spatially warped into the standard space).


**Goal:**
- *Presymptomatic prediction*: Classifying AD from healthy individuals is easy. Even non-deep learning methods can easily give more than  90% of accuracy. Also, this classification has minimal clinical utility (too late). What is clinically useful is to make a prediction of (future) risk for AD when an individual shows minimal to no symptoms of cognitive impairment. This prediction will have clinical impact because it will enable early intervention to delay the course of disease. 
- 


**Datasets:** 

- NACC (National Alzheimer Coordinating Center) has ~8000 MRI sessions each of which may have multiple runs of MRI. Some patients have longitudinal follow-ups. Patients and healthy controls. Clinical data (label data) is available. 
- OASIS (Open Access Series of Imaging Studies) has ~2000 MRI. Patients and healthy controls. Clinical data (label data) is available.


**Model selection:**

- 3D Convolutional Autoencoder: use this if needed.
- 3D Convolutional Neural Networks: the primary model with ReLU activation and Xavier initialization of filter parameter for each convolutional layer, max pooling method for the pooling layer, and softmax for the flattened layer.


**Accomplishments:**
1. 

**To-do list:**
1. 
