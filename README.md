# Project Proposal
### CS7641 Project Group #30 - Spring 2021

Clayton Horsfall, Seil Kwon, Connor O’Brien, Aidan Worswick, Andrew Yuchen Lin

### Introduction/Background
Alzheimer’s disease is currently the 6th leading cause of death in the United States. It is the most common type of dementia and more than 5 million Americans are living with Alzheimer’s disease. Timely diagnosis of dementia is crucial to allow patients to receive treatments for their symptoms, reduce patients’ anxiety, and save patients money by avoiding long-term medical costs. Alzheimer’s disease is typically diagnosed by highly specialized medical professionals after analyzing Magnetic Resonance Imaging (MRI) scans. This process is costly, and a machine learning approach to Alzheimer’s disease diagnosis from MRIs could greatly improve patient outcomes.
### Problem Definition
Construct a deep learning convolution neural network (CNN) model to detect Alzheimer’s disease, using multi-level and multimodal features of Alzheimer’s Disease Neuroimaging Initiative (ADNI) MRI brain scans.

### Methods
Our primary method will be a CNN, as they have achieved high classification accuracy in similar medical image diagnosis research. We will explore using both 2D and 3D CNN approaches, as well as using a two- track CNN approach (e.g., VGG19 and VGG16).

We will primarily use the Keras library with a TensorFlow backend to train the model. For our classifier, we will begin with supervised methods and an Extreme Learning Machine classification approach as it had proved highly accurate on similar problems. We will also investigate other classification procedures, such as Naïve Bayes, Support Vector Machines, SoftMax, and Ensemble Tree classifiers.

If an imbalance in the training data proves to be an issue, we will use data augmentation to train our CNN model and tune the pre-trained model using methods such as Generative Adversarial Networks.

We will initially develop our model with supervised methods, then work on an unsupervised model, as well as combine new methods such as few-shot learning and zero-shot learning. Finally, depending on the feasibility, we will deploy an app with our model that can classify Alzheimer’s disease from new MRI scans.

### Potential Results
Our model will use a CNN approach on 2D MRI images for the multi-modality classification of Alzheimer's Disease. The dataset contains 1075 patient MRI scans classified as having Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), or Cognitively Normal (CN). 

Wee et. al. achieved an AD classification accuracy of 92.35% in a similar Alzheimer’s disease classification problem, and we hope to match or improve upon their prediction accuracy.

### Discussion
Early diagnosis of Alzheimer's disease is extremely important for the initiation of effective treatment and future therapeutic development. While our model uses 2D images, AD diagnoses is generally a more complicated problem that requires 3D images, multiple cross sections, and complex data. However, we hope our preprocessing techniques, validation, and optimization strategies can establish groundwork for further exploration.

Another major limitation is our data size. Deep learning methodologies require large amounts of training data. While hybrid methods that combine traditional machine learning with deep learning approaches can be useful when there is limited data, ongoing collection of patients’ information in datasets such as ADNI is crucial to the academic research of this problem.

### References
Bangalore Yogananda CG, Shah BR, Vejdani-Jahromi M, et al. A Fully Automated Deep Learning Network for Brain Tumor Segmentation. Tomography. 2020;6(2):186-193. doi:10.18383/j.tom.2019.00026

Khan MA, Ashraf I, Alhaisoni M, et al. Multimodal Brain Tumor Classification Using Deep Learning and Robust Feature Selection: A Machine Learning Application for Radiologists. Diagnostics (Basel). 2020;10(8):565. Published 2020 Aug 6. doi:10.3390/diagnostics10080565

Oh, K., Chung, YC., Kim, K.W. et al. Classification and Visualization of Alzheimer’s Disease using Volumetric Convolutional Neural Network and Transfer Learning. Sci Rep 9, 18150 (2019). https://doi.org/10.1038/s41598-019-54548-6

Toprak A. Extreme Learning Machine (ELM)-Based Classification of Benign and Malignant Cells in Breast Cancer. Med Sci Monit. 2018;24:6537-6543. Published 2018 Sep 17. doi:10.12659/MSM.910520

Vemuri P, Wiste HJ, Weigand SD, et al. MRI and CSF biomarkers in normal, MCI, and AD subjects: diagnostic discrimination and cognitive correlations. Neurology. 2009;73(4):287-293. doi:10.1212/WNL.0b013e3181af79e5
Wee CY, Yap PT, Shen D. Prediction of Alzheimer’s disease and mild cognitive impairment using cortical morphological patterns. Hum. Brain Mapp. 2013;34:3411–3425. doi: 10.1002/hbm.22156. 
