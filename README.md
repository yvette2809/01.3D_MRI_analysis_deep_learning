# 3D_MRI_analysis_deep_learning


**Description:** 

About 10,000 brain structure MRI and their clinical phenotype data is available. Some MRI are longitudinal (each participant was followed up several times). MRI data has been preprocessed using standard brain imaging analysis pipeline (denoised, bias corrected, and spatially warped into the standard space).


**Datasets:** 

- NACC (National Alzheimer Coordinating Center) has ~8000 MRI sessions each of which may have multiple runs of MRI. Some patients have longitudinal follow-ups. Patients and healthy controls. Clinical data (label data) is available. 
- OASIS (Open Access Series of Imaging Studies) has ~2000 MRI. Patients and healthy controls. Clinical data (label data) is available.


**Model Selection:**

- 3D Convolutional Autoencoder: use this if needed.
- 3D Convolutional Neural Networks: the primary model with ReLU activation and Xavier initialization of filter parameter for each convolutional layer, and 


**To-do list:**

