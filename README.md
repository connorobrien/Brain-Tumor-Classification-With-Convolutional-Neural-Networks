# Midterm Report
## CS7641 Project Group #30 - Spring 2021
Clayton Horsfall, Seil Kwon, Connor O’Brien, Aidan Worswick, Andrew Yuchen Lin
## Introduction/Background
Alzheimer’s disease is currently the 6th leading cause of death in the United States. It is the most common type of dementia and more than 5 million Americans are living with Alzheimer’s disease. Timely diagnosis of dementia is crucial to allow patients to receive treatments for their symptoms, reduce patients’ anxiety, and save patients money by avoiding long-term medical costs. Alzheimer’s disease is typically diagnosed by highly specialized medical professionals after analyzing Magnetic Resonance Imaging (MRI) scans. This process is costly, and a machine learning approach to Alzheimer’s disease diagnosis from MRIs could greatly improve patient outcomes.

## Problem Definition
Construct a deep learning convolution neural network (CNN) model to detect Alzheimer’s disease, using multi-level and multimodal features of Alzheimer’s Disease Neuroimaging Initiative (ADNI) MRI brain scans. 

##  Data Collection
The data set collection (provided by ADNI) includes 199 baseline 3T MRI scans, each classified as cognitively normal (CN), mild cognitive impairment (MCI), or Alzheimer’s Disease (AD). The data also include demographic variables such as the subject’s age and sex, as well as the date the scan was taken. The MRI images were provided in a NIfTI format, which we converted to 3D arrays using [NiBabel](https://nipy.org/nibabel/).

## Methods

The MRI data had been uniformly preprocessed by ADNI using Gradwarp, B1 non-uniformity, N3, and scaling. These procedures, taken from Wyman et al., 2013, are as follows:

1.	Gradwarp: Gradwarp is a system-specific correction of image geometry distortion due to gradient non-linearity. The degree to which images are distorted due to gradient non-linearity varies with each specific gradient model. We anticipate that most users will prefer to use images which have been corrected for gradient non-linearity distortion in analyses.
2.	B1 non-uniformity: this correction procedure employs the B1 calibration scans noted in the protocol above to correct the image intensity non-uniformity that results when RF transmission is performed with a more uniform body coil while reception is performed with a less uniform head coil.
3.	N3: N3 is a histogram peak sharpening algorithm that is applied to all images. It is applied after grad warp and after B1 correction for systems on which these two correction steps are performed. N3 will reduce intensity non-uniformity due to the wave or the dielectric effect at 3T. 1.5T scans also undergo N3 processing to reduce residual intensity non-uniformity.

We subsequently normalized the image data, fixed the orientation by rotating the volumes by 90 degrees, and resized the volumes so they are at the same width (256,256,166). Finally, due to the small sample size, we used data augmentation to artificially expand the sample size. Our current data augmentation involves slightly rotating (-20 to 20) the volumes. As will be discussed later, we seek to implement Generative Adversarial Network for image synthesis to increase the training data size.

Our primary supervised method is a convolution neural network (CNN), as they have achieved high classification accuracy in similar medical image diagnosis research. We utilized the Keras library with a TensorFlow backend to train the CNN model. Our current approach is based on a CNN model developed by Zunair et al., 2020, utilizing convolution layers, max and average pooling, batch normalization, dense layers, and dropout layers. Our CNN model concludes with a binary SoftMax classifier. 

<p align="center">
  <img src='https://i.imgur.com/1YEUN2w.png' width="500">
</p>

The model was compiled with a binary cross entropy loss function and optimized using the Adam algorithm introduced by Kingma et al., 2014. Our Adam algorithm has an initial learning rate of 0.001 with a custom exponential decay schedule.

## Results
Our final model achieved a 67% classification accuracy with a ROC area under curve of 69%. The model accuracy and loss for the training and validation sets are plotted below.

...adding image here later...


Further, our model can predict the class probabilities for MRI scans. The predicted probabilities for the MRI scan pictured below are 54% cognitively normal and 46% Alzheimer’s diseases. 
 
 <p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Deep-Learning-for-Alzheimers-Disease-Detection/main/Images/example_mri_scan.png?token=ASLEFKPUOLC2VVHS4FO4YXLAPCGGKE' width="500">
</p>
 

Wee et. al. achieved an AD classification accuracy of 92.35% in a similar Alzheimer’s disease classification problem, and in the following weeks we hope to match or improve upon their prediction accuracy.

## Discussion
As we progress towards our final report, there are several areas where we will look to improve our model and add additional features. First, we will experiment with our CNN layers and parameters to improve our model’s classification accuracy. We will also look to incorporate multiclass classification that includes patients with mild cognitive impairment, along with those classified as cognitively normal and having Alzheimer’s disease.

Further, in an attempt to overcome our small sample size, we will experiment with different Generative Adversarial Networks (GANs) for image synthesis. Yi et al, 2019, surveyed 150 published articles using adversarial training schemes in medical imaging. Common GAN medical image reconstruction approaches include pix2pix, CycleGan, and SGAN, common loss measures adversarial loss and element wise fidelity loss, and common quantitative measures include normalized MSE with respect to ground trough, and peak signal to noise ratio with respect to ground truth. We will aim to incorporate at least one GAN image synthesis approach to artificially increase the size of the training data. 

Early diagnosis of Alzheimer's disease is extremely important for the initiation of effective treatment and future therapeutic development. We hope our preprocessing techniques, validation, and optimization strategies can establish groundwork for further exploration. While hybrid methods that combine traditional machine learning with deep learning approaches can be useful when there is limited data, ongoing collection of patients’ information in datasets such as ADNI is crucial to the academic research of this problem.

## References
Bangalore Yogananda CG, Shah BR, Vejdani-Jahromi M, et al. A Fully Automated Deep Learning Network for Brain Tumor Segmentation. Tomography. 2020;6(2):186-193. doi:10.18383/j.tom.2019.00026

Hasib Zunair, Aimon Rahman, Nabeel Mohammed, & Joseph Paul Cohen. (2020). Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction.

Kingma, Diederik P. et al. "Adam: A Method for Stochastic Optimization." (2017).

Khan MA, Ashraf I, Alhaisoni M, et al. Multimodal Brain Tumor Classification Using Deep Learning and Robust Feature Selection: A Machine Learning Application for Radiologists. Diagnostics (Basel). 2020;10(8):565. Published 2020 Aug 6. doi:10.3390/diagnostics10080565

Oh, K., Chung, YC., Kim, K.W. et al. Classification and Visualization of Alzheimer’s Disease using Volumetric Convolutional Neural Network and Transfer Learning. Sci Rep 9, 18150 (2019). https://doi.org/10.1038/s41598-019-54548-6

Toprak A. Extreme Learning Machine (ELM)-Based Classification of Benign and Malignant Cells in Breast Cancer. Med Sci Monit. 2018;24:6537-6543. Published 2018 Sep 17. doi:10.12659/MSM.910520

Vemuri P, Wiste HJ, Weigand SD, et al. MRI and CSF biomarkers in normal, MCI, and AD subjects: diagnostic discrimination and cognitive correlations. Neurology. 2009;73(4):287-293. doi:10.1212/WNL.0b013e3181af79e5 Wee CY, Yap PT, Shen D. Prediction of Alzheimer’s disease and mild cognitive impairment using cortical morphological patterns. Hum. Brain Mapp. 2013;34:3411–3425. doi: 10.1002/hbm.22156.

Wyman, B. T., Harvey, D. J., Crawford, K., Bernstein, M. A., Carmichael, O., Cole, P. E., Crane, P. K., DeCarli, C., Fox, N. C., Gunter, J. L., Hill, D., Killiany, R. J., Pachai, C., Schwarz, A. J., Schuff, N., Senjem, M. L., Suhy, J., Thompson, P. M., Weiner, M., Jack, C. R., Jr, … Alzheimer’s Disease Neuroimaging Initiative (2013). Standardization of analysis sets for reporting results from ADNI MRI data. Alzheimer's & dementia : the journal of the Alzheimer's Association, 9(3), 332–337. https://doi.org/10.1016/j.jalz.2012.06.004

Xin Yi, Ekta Walia, & Paul Babyn (2019). Generative adversarial network in medical imaging: A review. Medical Image Analysis, 58, 101552.
