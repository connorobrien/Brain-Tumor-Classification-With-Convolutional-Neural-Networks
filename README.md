# Final Report
Clayton Horsfall, Seil Kwon, Connor O’Brien, Aidan Worswick, Andrew Yuchen Lin
## 1. Introduction/Background

The National Brain Tumor Society estimates there are currently 700,000 people in the United States living with a primary brain tumor, which are masses of abnormal cells in the brain. Many types of brain tumors exist, and they can be either cancerous or benign. There is large variance in the severity of brain tumors, with the average 5-year survival rates ranging from 36% for cancerous brain tumors to 91.7% for benign brain tumors.  

Three of the most prominent types of brain tumors are glioma, meningioma, and pituitary tumors. Gliomas, which account for around 33% of brain tumors, may be life threatening based on their size and location. A patient diagnosed with a glioma will require an aggressive treatment plan that could include surgery, radiation therapy, and/or chemotherapy. 

Meningiomas are common tumors of the brain and spinal cord. Most meningiomas are benign, with only around 10% of them being cancerous. Benign meningiomas are treated by monitoring for symptoms, and they may be removed surgically if symptoms arrive. Cancerous meningiomas are typically treated with radiation therapy. 

Pituitary tumors are abnormal growths that occur in the pituitary gland of the brain and are commonly benign. Pituitary tumors follow a similar treatment plan to meningiomas, where observation and/or surgery is most common and radiation therapy is typically used to treat cancerous tumors.  

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/different_tumor_types.png?token=ASLEFKNFMICL2ZHTHY7S2ZTASHJI2' width="600">
</p>

Due to the high variance in the risk level of these three types of brain tumors, an early and accurate diagnosis of the specific type of brain tumor is crucial to allow patients to receive timely treatment and an appropriate plan of care. Diagnosis of brain tumors typically begins with Magnetic Resonance Imaging (MRI) scans to observe that a tumor is present. After an MRI scan shows a patient has a brain, doctors will determine the type of brain tumor by conducting a biopsy or surgery. This process has large financial and time costs, and a machine learning approach to brain tumor classification from MRIs could greatly improve patient outcomes. 

## 2. Problem Definition
Construct a deep learning convolution neural network (CNN) model to classify brain tumors as gliomas, meningiomas, and pituitary tumors, using 2D MRI brain scans. 

## 3. Data Collection

The data set was collected from two hospitals in China (Nanfang Hospital and General Hospital, Tianjing Medical University) between 2005 and 2010. It includes 3064 baseline T1-CE MRI scans of patients diagnosed with one of the three brain tumors discussed above. There are 1,426 images of patients with gliomas, 708 images of patients with meningioma, and 930 images of patients with pituitary tumors. Each image is 512 x 512 pixels in size. The grey scale MRI images were provided in .mat format, which we converted to 2D arrays using the mat73 Python package. 


## 4. Literature Survey
### 4.1 Brain tumor classification using deep CNN features via transfer learning, March 2019 

Computer Aided Diagnosis (CAD) is a relatively new medical application in which large volumes of data, particularly in the form of imagery, can be fed to a classification algorithm, yielding a level of certainty about the presence of brain tumors, tumor classification, brain atrophy, cognition loss, and more. This paper focuses on classifying brain tumors into three groups: glioma, meningioma, and pituitary. These three prominent types of brain tumor are sufficiently different in growth and treatment that accurate classification is critical to ensuring the right kind of care and treatment are provided. 

The proposed system in this paper uses a pretrained GoogLeNet for feature extraction from brain imagery, using the same dataset we used for our CNN model, and yields a reported best-in-class mean classification accuracy of 98%. The paper posits that that transfer learning is a useful classification technique when data is limited. 

Brain tumor subtype classification has seen rapid advancement in the last six years, largely due to high-contrast MRI advantages over traditional computed tomography (CT) images for soft tissue in the brain, as well as extensive study on deep-learning approaches. Because brain tumors exhibit high variation in size, shape, and intensity, tumors across classifications may show similar appearance on an MRI, presenting a large challenge for improving upon existing models. The primary advantage of this paper’s proposed transfer learned CNN model is that it can provide considerably “good” performance even when there is a small training data set. 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/googlenet.jpg?token=ASLEFKPKRS26TH7FB7XEF23ASHJL6' width="600">
</p>
  
The GoogLeNet network, the knowledge from which the authors transferred to their CNN, is a 22-layer network which has been recognized as the top-performing image-recognition network in the 2014 Imagenet Large Scale Visual Recognition Challenge. To augment the GoogLeNet network to the three-type classification problem for brain tumors, the authors modified the last three layers of GoogLeNet to adapt it to the MRI dataset. They substituted the Fully Connected (FC) layer with a new one that had an output size of three, and similarly replaced the softmax layer and the cross-entropy-based classification layer. 

In pre-processing, the MRIs were normalized in intensity between 0 and 1, and resized from 512x512 to 224x224, which aligned with GoogLeNet’s original input-size. They also used Adam as their optimizer, which we will also utilize in our CNN as it is generally accepted to have great performance per computational requirement. Their initial learning rate was 0.0003. Using accuracy as their measure (correctly classified samples, divided by total samples), they achieved 97.8% accuracy with a Support Vector Machine (SVM) classifier on their CNN features, and a 98.0% accuracy with a K-Nearest Neighbors (KNN) classifier. 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/googlenet2.jpg?token=ASLEFKPZFRNTCGWW4R4NTM3ASHJOU' width="500">
</p>
  
 The authors found that most misclassifications pertained to the meningioma tumor type, citing that this was the smallest class in the data sample, and that no specific data augmentation was used to balance the dataset either. They cite future work as using data augmentation to balance this, as they noted overfitting when they decreased the training size by 75%.

### 4.2 Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and 	Partition, October 2015 

This paper is one of the earlier applications of CNN models in brain imagery, using the same dataset that we will analyze in our application. Most papers on the topic of brain tumor classification using MRI and CAD cite this paper as a benchmark for performance, where an accuracy of 84-88% was achieved using an intensity histogram, gray level co-occurrence matrix (GLCM), and bag-of-words (BoW) models. 

This paper focuses on Tumor Region Augmentation, in which the authors note the subregion in which the tumor exists but dilate the borders of those subregions in the image, citing that the tissues surrounding the tumor can offer important information on the type-classification. The region of interest (ROI), having been augmented to discriminate between classes, dramatically improves the performance of the model, as a ring-form partition allows features in each subregion to be extracted separately. 

A major driver for this ROI expansion and augmentation is the previously-noted characteristic that tumor-surrounding tissue can lend valuable insight to the discrimination of type: meningiomas are typically adjacent to skull fluid, gray matter, and cerebrospinal fluid; gliomas are associated with white matter; and pituitary tumors are often adjacent to sphenoidal sinus (the area behind the nose and between the eyes responsible for mucus production), internal carotid arteries (blood flow connections via the neck), and optic chiasma (crossing of the optic nerves in the brain). 

The authors found that their BoW approach, which utilized an image patch as a local feature and considers relationships between multiple neighboring image pixels, strongly outperformed the intensity histogram (which uses single-pixel) and GLCM (which uses pairwise pixel relationships). 

We will similarly use this model’s accuracy as a baseline for our CNN approach, though we will aim to achieve accuracies more in the realm of those achieved by expansions to this paper’s methods. Given the rapid growth in this subject, an accuracy of 95% should be our goal for our initial CNN model, which will then be compared to other pre-trained models like UNet and ResNet. 

### 4.3 Multi-grade brain tumor classification using deep CNN with extensive data augmentation, July 2018 

The authors in this paper address one of the most common issues with CNN training, which is data availability and data-set size. They do so by expanding on previous deep CNN brain tumor classifications, while adding extensive data augmentation to work around the lack of data problem. We encountered this very problem in our initial attempt at Cognitive Atrophy, using 3D MRIs to classify the presence of Alzheimer’s Disease or mild cognitive impairment. Given the larger size of the image matrices, our dataset was smaller in quantity, and in training we observed no improvements to loss, accuracy, or AUC, as the classification was unable to pick up any important features. We attempted augmentation to expand the original dataset but found our methods inefficient. 

The focus on the tumors themselves in this analysis is centered on the grade of the tumor, rather than type-classification (our focus). The World Health Organization classifies central nervous system tumors into malignancy grades from I to IV (benign to high-malignancy)—glioblastomas are considered the most lethal type of tumor, considered Grade IV by the WHO, carry devastating prognoses despite modern medical advancements. Grade-classification is currently done by pathologists performing histopathology, measuring the necrosis, microvascular proliferation, and vascular thrombosis (blood clotting) of the cancer. These features do not necessarily have hard-and-fast thresholds, and pathologists have been observed grading tumors differently. MRIs allow for non-surgical evaluation of the tumor, but without the ability to “touch” the tissue, medical professionals must turn their reliance to CAD for classification systems. 

The approach, which claims to be the first attempt at CAD implementation on WHO standards for cancer grade classification, works as follows: the brain tumor is segmented from the MRO, the segmented region is augmented with various techniques, and a classification is made into one of four grades. All convolutional layers in their CNN model use 3x3 kernels on a single stride, including the initial stride. 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/augmentation.jpg?token=ASLEFKJHNB2ZPOBI6L6UMFLASHJP2' width="500">
</p>
  
The primary focus of this paper, as previously stated, is the data-augmentation section. To build a more robust data set for their deep learning, the authors performed eight different augmentation techniques: rotation, flipping, skewness, shearing, Gaussian blur, sharpening, edge detection, and emboss. Within these techniques, they utilize 30 different parameters to extend each sample of data into 30 samples. 

They use the VGG-19 CNN architecture for grade classification, which consists of 19 weighted layers (16 convolutional and 3 fully connected). Again, this architecture was chosen for its consistent use of 3x3 kernels on single strides exclusively throughout the convolutions. 

This paper also uses the same brain tumor dataset we will use in our analysis, and their data-augmentation method increases their sample size from 3064 images to 91,920. Before augmentation, their CNN model achieved results of 84.51% sensitivity and 93.34% specificity. After augmentation, they observed lifts to these measurements to 88.41% (+3.9 pts) and 96.12% (+2.78 pts). 

Conclusively, this paper posits that expanding the data set with parameter adjustments can improve model performance. Their accuracy post-augmentation of 94.58% is in-line with, if not better than, current state-of-the-art models, though they concede that future work would involve more focus on the deep learning model (i.e., something other than VGG19, or perhaps more tuning to it). They further acknowledge the cost-benefit of computational needs of running a model on 3000 versus 90,000 images and propose a lighter-weight CNN to ease those computational constraints. 

## 5. Methods
For this project, we implemented several 2D CNN architectures to classify different types of brain tumors. Previous studies using CNNs with brain MRIs for medical diagnoses have used either 2D or 3D images. With 3D images, the models can take advantage of additional context to improve model performance. However, this additional context comes at a large computational cost as training a CNN with 3D images requires far more parameters to be estimated. Several preprocessing methods have been proposed to reduce the computational cost of using 3D MRI images, while still maintaining the additional context the images provide when compared to using 2D MR images. 

Although our initial plan was to use these methods and build the CNN model using 3D data, there were several challenges that directed us to alternatively use the 2D CNN model for this project. While each 2D image was trained on resolution of 512 x 512 pixels with size of 200KB, 3D images were trained on high resolution of 256 x 256 x 170 with size that exceeded 40MB per image. Because of the large 3D data size, the speed of training our 3D model became an issue. While training iterations for the 2D CNN finished within a day, it may have taken over weeks to train the 3D CNN models on our GPU if we increased the epochs, which in turn made it hard to tune the parameters in time. Moreover, while the 3D patches had a depth of 166, the 3D CNN had a much smaller output stride, which caused complications with the max pooling layers. For these reasons, we built a 2D CNN model for our experiment. 
### 5.1 Primary CNN Model 

We developed our baseline CNN model using insights gained from research on similar problems. We used a combination of the Keras library with a TensorFlow backend along with Pytorch. Our primary CNN approach uses convolution layers, max pooling, batch normalization, dense layers, dropout layers, and a categorical SoftMax classifier. The full model architecture is depicted below. 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/Primary_model_architecture.jpg?token=ASLEFKKRWVFYBEE2DJN6PZLASHJR4' width="600">
</p>

We experimented with various architectures and found this model performed well at classifying brain tumors in our data set. In subsequent sections, we compare this approach with other well-known CNN architectures. This CNN model has 30 layers and 33,774,340 parameters to be estimated. 

The model classifies the brain tumors into three groups: glioma, meningioma, and pituitary tumor. Since we are classifying images into more than just two groups, we applied a categorical cross-entropy loss function with one-hot encoded labels. Essentially, this loss function, which takes the negative log of the predicted probability of an outcome, behaves in a such a way that as the predicted probability of the assigned (“1” or predicted) class gets close to zero, the loss increases exponentially. Taking the mean of all the losses of each prediction yields the categorical cross-entropy, and this is precisely what we are trying to minimize. 

The model was optimized using the Adam algorithm introduced by Kingma et al., 2014. Our Adam algorithm has an initial learning rate of 0.001 with a custom exponential decay schedule. The Adam algorithm is considered to be well-suited for deep learning, natural language processing, and, in our case, computer vision. It further has benefits for our use-case, as it is easy to implement and is computationally efficient. Adam combines features of momentum and adaptive learning rate algorithms to converge quickly without a loss of accuracy. 

### 5.2 ResNet 

ResNet (residual neural network) is a set of neural network architectures that address the vanishing gradient problem by introducing short-cut connections. CNNs that are deep with many layers can face the vanishing gradient problem where the gradient becomes too small for the weights to update effectively, and the model cannot learn on the given dataset. In this situation, the accuracy saturates and degrades quickly. ResNet uses residual blocks and short-cut connections that bypass certain stacked layers with an identity mapping. 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/resnet.jpg?token=ASLEFKJR3OJNME7V2XN5MYLASHJT4' width="500">
</p>

The above diagram shows how these short-cut connections work. CNN layers typically map an input x to an output G(x). Since the direct mapping of x to G(x) can be difficult to learn due the vanishing gradient, we can instead learn the direct mapping of the residual, F(x), where F(x) = G(x) – x. Above, we have a layer learn F(x) and then have an identity mapping to add x to it. This achieves the same G(x) that we are interested in, but we do not need to worry about the gradient becoming too small. This identity mapping solves the vanishing gradient problem by allowing the gradient to flow through this alternate path. ResNets have been shown to be highly effective at training extremely deep neural networks. 

In this project, we experiment we three different variations of ResNet; namely, ResNet18, ResNet50, and ResNet101. The number at the end of each variation’s name represents the number of layers in the residual neural network. When determining the number of layers to include in a deep CNN, we face a trade-off between accuracy and computational complexity. As we increase the number of layers, the model estimates many more parameters, allowing the CNN to accurately fit to more complex relationship in the data. This increased accuracy comes at the cost of requiring more computational power to estimate these parameters. There is also increased risk of overfitting to the training data as we increase the number of layers.  


### 5.3 Few-shot Learning  

Few-shot classification is a semi-supervised learning problem in which a classifier must be adapted to accommodate new classes not seen in training, given only a few examples of each of these classes. A naïve approach, such as retraining the model on the new data, would severely overfit. While the problem is quite difficult, it has been demonstrated that the algorithm has ability to perform even one-shot classification, where only a single example of each new class is given with a high degree of accuracy. 

In our experiment, we implemented prototypical networks for the problem of few-shot classification, where a classifier must generalize to new classes, given only a small number of examples of each new class. Prototypical networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class. Compared to recent approaches for few-shot learning, they reflect a simpler inductive bias that is beneficial in this limited-data environment and achieve excellent results. 


<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/fewshot.jpg?token=ASLEFKNAXJKIVVRMD5D4GXLASHJVE' width="400">
</p>
As shown above, (a) Few-shot prototypes  are computed as the mean of embedded support examples for each class. (b) Zero-shot prototypes  are produced by embedding class meta-data . In either case, embedded query points are classified via a softmax over distances to class prototypes: . 

### 5.4 Data Augmentation via Unsupervised Learning (GAN) 

An effective CNN model requires a large amount of data to train. The more available quality input data, the better the CNN can be trained as a useful classifier. Yet, a major problem in the field of medical imaging is the data sets are often imbalanced, as abnormal findings are rare compared to normal ones. Another major issue is the restrictions around privacy and using patient data. Traditional data augmentation techniques, such as crops, flip, translation, rotation, are designed to tackle these challenges faced in data collection. However, the products created by these techniques are fundamentally highly correlated. 

Therefore, to increase the sample size of our training data, we experimented with different Generative Adversarial Networks (GANs) for image synthesis. Yi et al, 2019, surveyed 150 published articles using adversarial training schemes in medical imaging. Common GAN medical image reconstruction approaches include pix2pix, CycleGan, and SGAN. Further, common loss measures adversarial loss and element wise fidelity loss, and common quantitative measures include normalized MSE with respect to ground trough and peak signal to noise ratio with respect to ground truth. 

GAN is an unsupervised machine learning technique; the vanilla idea is that we simultaneously train two CNN models that competes against each other. The generator CNN is a forger trying to produce realistic counterfeits, and the detector CNN is a police officer trying to detect the fakes. We hoped GAN could produce useful tumor MRI resemblance, and greatly expand our training data size. Our basic GAN architecture consists of 3 layers of simple linear with accommodating ReLu activators and ends with a Tanh activator for the generator and a Sigmoid activator for the detector. 
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/sigmoid.png?token=ASLEFKJVCLMYXRC6CJLBRSDASHJV4' width="400">
</p>
Unfortunately, the results from our vanilla GAN model were too low in resolution to be used as input to our CNN models, but it is indeed an exciting field that many researchers are looking into and making breakthroughs. Below are examples of the input and output for our GAN model. 

**Input Images:** 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/gan_input.png?token=ASLEFKNZKYG7D6MT7PEX3I3ASHJXG' width="400">
</p>

**Output Images:** 

<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/gan_brains.png?token=ASLEFKJAITR3LCRD6R4EAX3ASHJXY' width="400">
</p>
To verify our GAN model, we changed our input to human faces, and the results are more intuitive to evaluate. Although very promising, the output quality is similarly below acceptable. 
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/gan_faces.png?token=ASLEFKMA3PCNEDBKICILKFLASHJYO' width="400">
</p>


## 6. Results 

Our most accurate model was the one based off the ResNet18 architecture. This model had a perfect training accuracy and AUC, and a validation accuracy of 93.5% and AUC of 99.26%. AUC measures how well the algorithm is able distinguish between classes. A value of 0.5 means the model is no better than random, and a value of 1 means the model can perfectly separate the data into their correct classes.    
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/table_model_accuracy.png?token=ASLEFKLNLL3STSWLGLTZ6HLASHJ2Y' width="400">
</p>

The ResNet18 model was the most accurate, and it also included considerably less parameters than the other models. However, the difference across all four models AUC values is relatively small. Further, though the ResNet18 model yielded the highest prediction accuracy, our primary CNN model required considerably less epochs to before the model accuracy stabilized. Epochs define how many times the model will update its parameters by working through the training dataset. Below are the loss and AUC plots across for our primary CNN model and the ResNet18 model. The ResNet50 and ResNet101 plots are similar to the ResNet18’s. 

**Primary CNN AUC and Loss:**
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/Primary_cnn_model_AUC.jpg?token=ASLEFKLEKNXSUSU4HDU4NA3ASHJ3M' width="400">
</p>

**ResNet18 CNN AUC and Loss:**
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/ResNet18_AUC.jpg?token=ASLEFKLYAVAJUD5V6DBJGA3ASHJ34' width="400">
</p>

Although our semi-supervised Few-shot learning model used a small number of training data for each class, it still had accuracy of 0.89 using 3 samples and 0.87 accuracy using only 1 sample. N-way represents number of classes, and episode is the training iteration in the context of Few-shot Learning. It is a step in which we train the network, calculate loss, and backpropagate the error. For our experiment, we set our episodes to 20,000 and frame-size at 1,000. As shown below, it is remarkable that increase in training sample size is not proportional to increase in accuracy and that 3-shot learning achieved higher accuracy than our ResNet50 model. 
<p align="center">
  <img src='https://raw.githubusercontent.com/connorobrienedu/Brain-Tumor-Classification-With-Convolutional-Neural-Networks/main/Images/table_fewshot.png?token=ASLEFKORLXMWR4GZLHXMHOTASHJ4Q' width="400">
</p>

## 7. Discussion
Deepak and Ameer, with their transfer-learning model applied to the same brain tumor dataset, achieved 92.3% accuracy on their baseline model, 97.8% with their SVM classifier, and 98.0% accuracy with their KNN classifier. Their full SVM classifier model produced AUC values of 99.9%, 99.7%, and 99.5% for Gliomas, Pituitaries, and Meningiomas respectively. Cheng et al. obtained an overall accuracy of 92.18% using their expanded tumor region dilation and augmentation method, with a sensitivity and specificity of 81% and 92%, respectively. Sajjad et al. achieved a sensitivity, specificity, and accuracy of 88.41%, 96.12%, and 94.58% with their 30-parameter data augmentation approach. 

With our custom CNN architecture achieving a 90.83% validation accuracy, our best ResNet model achieving a 93.05% validation accuracy, and our best few-shot model achieving an 89.63% prediction accuracy, our approaches highlight methods that didn’t quite perform to the state-of-the-art benchmarks, as well as methods that met those benchmarks. 

Moreover, although Prototypical Networks produced great results despite using only few samples of data, they still have limitations. First, they lack generalization. They produced an accuracy close to 99% on dataset like Omniglot because handwritten characters share similar characteristics; however, as shown from our experiment, accuracies decrease on complex data like medical images. Another limitation is prototypical networks only use mean to decide the center, ignoring the variance in support set. Therefore, we can also consider using other Few-shot learning methods such as Gaussian Prototypical Networks or Triplet loss for more complicated tasks. 

We see room to improve upon our models in a few areas. While the dataset was reasonably large, we would have liked to incorporate degrees of data augmentation to artificially inflate the training data. Several studies (Wyman et. al.) have shown data augmentation approaches that have been successful in the medical image classification field. Further, we utilized a GPU for our models, but still encountered memory errors. If we were able to access more RAM or improve the efficiency of our CNN models, we may have been able to further improve the model performance. 

## References

Bae, J.B., Lee, S., Jung, W. et al. Identification of Alzheimer's disease using a convolutional neural network model based on T1-weighted magnetic resonance imaging. Sci Rep 10, 22252 (2020). https://doi.org/10.1038/s41598-020-79243-9 

Bangalore Yogananda CG, Shah BR, Vejdani-Jahromi M, et al. A Fully Automated Deep Learning Network for Brain Tumor Segmentation. Tomography. 2020;6(2):186-193. doi:10.18383/j.tom.2019.00026 

Cheng, Jun, et al. “Correction: Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition.” PLOS ONE, vol. 10, no. 12, 2015, doi:10.1371/journal.pone.0144479. 

Deepak, S., and P.M. Ameer. “Brain Tumor Classification Using Deep CNN Features via Transfer Learning.” Computers in Biology and Medicine, vol. 111, 2019, p. 103345., doi:10.1016/j.compbiomed.2019.103345. 

“Gliomas.” Johns Hopkins Medicine, www.hopkinsmedicine.org/health/conditions-and-diseases/gliomas  

Hasib Zunair, Aimon Rahman, Nabeel Mohammed, & Joseph Paul Cohen. (2020). Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction. 

He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, doi:10.1109/cvpr.2016.90 

Khan MA, Ashraf I, Alhaisoni M, et al. Multimodal Brain Tumor Classification Using Deep Learning and Robust Feature Selection: A Machine Learning Application for Radiologists. Diagnostics (Basel). 2020;10(8):565. Published 2020 Aug 6. doi:10.3390/diagnostics10080565 

Kingma, Diederik P. et al. "Adam: A Method for Stochastic Optimization." (2017). 

“Meningioma.” Mayo Clinic, Mayo Foundation for Medical Education and Research, 21 Apr. 2020, www.mayoclinic.org/diseases-conditions/meningioma/symptoms-causes/syc-20355643 

“Pituitary Tumors.” Mayo Clinic, Mayo Foundation for Medical Education and Research, 31 Mar. 2021, www.mayoclinic.org/diseases-conditions/pituitary-tumors/diagnosis-treatment/drc-20350553  

Puch, S., Sánchez, I., & Rowe, M. (2019, August 27). Few-shot Learning with Deep Triplet Networks for Brain Imaging Modality Recognition. arXiv.org. https://arxiv.org/abs/1908.10266. 

Sajjad, Muhammad, et al. “Multi-Grade Brain Tumor Classification Using Deep CNN with Extensive Data Augmentation.” Journal of Computational Science, vol. 30, 2019, pp. 174–182., doi:10.1016/j.jocs.2018.12.003. 

Snell, J., Swersky, K., & Zemel, R. S. (2017, June 19). Prototypical Networks for Few-shot Learning. https://arxiv.org/pdf/1703.05175.pdf. 

Toprak A. Extreme Learning Machine (ELM)-Based Classification of Benign and Malignant Cells in Breast Cancer. Med Sci Monit. 2018;24:6537-6543. Published 2018 Sep 17. doi:10.12659/MSM.910520 

Wyman, B. T., Harvey, D. J., Crawford, K., Bernstein, M. A., Carmichael, O., Cole, P. E., Crane, P. K., DeCarli, C., Fox, N. C., Gunter, J. L., Hill, D., Killiany, R. J., Pachai, C., Schwarz, A. J., Schuff, N., Xin Yi, Ekta Walia, & Paul Babyn (2019). Generative adversarial network in medical imaging: A review. Medical Image Analysis, 58, 101552. 
