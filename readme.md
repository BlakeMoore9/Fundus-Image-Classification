### Machine Learning Applications in Healthcare
# Deep Learning for Fundus Image Classification


![](images/BunteMischung.png)

<br>


# Contents
1. [Introduction](#introduction)
2. [Method](#method)
3. [Tools](#tools)
4. [Dataset and Exploratory Data Analysis](#dataset-and-exploratory-data-analysis)
5. [Tasks](#tasks)
6. [Preprocessing](#preprocessing)
7. [Augmentation](#augmentation)
8. [Anomaly Detection](#anomaly-detection)
9. [Multi-Class and Multi-Label Classificationation](#multi-class-and-multi-label-classification)
10. [Results](#results-and-conclusion)
11. [What comes next](#what-comes-next)
12. [References](#references)
<br>
<br>



# Introduction and Method
In this project, deep learning methods were used to predict abnormalities in fundus images from Kaggle. The images in the dataset were augmented and different colorschemes were tried. 

Aproach
- Collection of information on the field of Funduns Images,Illnesses and Epidemiology, for Benchmarking and Orientation
- Decision on tasks, metrics, tools and methods
- Exploration of dataset
- Preprocessing of data
- Augmentation of data
- Training of Model on various variants of architecture, hyperparameters and data
- Summarize results
<br>
<br>

# Tools 
For the exploration, preprocessing and deep learning application a variety of different tools were used. Some of the work was done locally on a desktop computer, more complex applications were carried out in the cloud. Herefore I mostly used Google Colab and Paperspace, until I ran into some memory issues and switched to Google CLoud Platform.

- Cloud Computing: Google Colab, Paperspace, Google Cloud Platform
- Data Exploration, Manipulation and Visualization: Python, NumPy, Pandas, Matplotlib
- Preprocessing & Augmentation: OpenCV, PIL, Pillow, Scikit-Image, KerasImageGenerator
- Deep Learning: Keras, TensorFlow


<br>
<br>

# Dataset and Exploratory Data Analysis
The dataset includes 3,285 images from CTEH (3.210 abnormals and 75 normals) and 500 normal images from Messidor and EYEPACS dataset. The abnormalities include: opacity, diabetic retinopathy, glaucoma, macular edema, macular degeneration, and retinal vascular occlusion.
<br>

Source<br>
https://www.kaggle.com/c/vietai-advance-retinal-disease-detection-2020/data
<br>

## Diabetic Retinopathy
Diabetic Retinopathy affects the blood vessels in the retina and is the leading cause of vision impairment and blindness. Important features for detection are Blood vessels, exudates, hemorrhages, micro aneurysms.
<br>

## Opacity
Corneal opacity is a disorder of the cornea which is the transparent structure on the front of the eyeball. Corneal opacity occurs when the cornea becomes scarred. This stops light from passing through the cornea to the retina and may cause the cornea to appear white or clouded over.
<br>

## Glaucoma
Glaucoma is caused by a fluid buildup in the eye, which causes an increase in eye pressure that damages the optic nerve.
<br>

## Macular Edema
A Macular Edema is caused by a collection of fluid deposits on or under the macula. This will make the macula thicken and swell. It is often associated with diabetes but can also be caused by age-related macular degeneration. 
<br>

## Macular Degeneration
Degeneration of the macula. Is mostly age-related, but the risk will be higher, if patient is a smoker or has a high cholesterol. 
<br>

## Retinal Vascular Occlusion
Retinal vein occlusions occur, when there is a blockage of veins of the nerve cells in the retina. 
<br>
<br>


# Preprocessing
The goal of preprocessing is preparing the data for modeling and make important features pop out. 
I started with cropping and resizing the images and then tried some variants of coloring and lighting. 

Color Variants
•	Original Color
•	Greyscale
•	Preprocessing for VGG-16
•	Substracting local average color
<br>
<br>

# Augmentation
The augmentation will be used to produce more data and help adressing class imbalances. 
I first tried to do this by only balancing the instances with merely one abnormality in the image. 

After running into various issues with memory while transforming the images to arrays with NumPy, I decided to go with the Keras Image Generator, were the images were transformed and rescaled on the fly during training. 
While I still had to balance the data manually, it was perfect for reading in images in a memory-friendly way. 
<br>
<br>

# Anomaly Detection, Multi-Class and Multi-Label Prediction
To predict anomalies in the images, I started off by using a simple CNN similar to VGG-16, a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. 
Next, I wanted to see, if I could predict multiple classes. I selected the four most prevalent illnesses in dataset. As the simple CNN did not show great results, I used a pretrained model, the aforementioned VGG-16 and did some fine tuning. 

![](images/vgg16_modified.png)
*Source: https://blog.keras.io/img/imgclf/vgg16_modified.png*

<br>
<br>

# Results and what comes next
To predict an anomaly from fundus images is easy to solve, but multi-class and multi-label predictions proved to be more complex. Next, I will try:

1.	A more data centric approach by using more data from different sources and try more preprocessing approaches. 
2. Addressing not only the imbalance in cases but also in labels.
<br>
<br>

# Thank you!
Mia, Eike, Simon, Anne, Torsten, Chris, Karl, Julia, Boris, Rainer, Juan, Adrian, Burak, Fidel, Matthias, Mario, Christian, Jaouad, Yusuf! I wish you all the best!  
<br>
I learned so much and had a great time!
Now, I will go on... and learn some more ;)

![](https://i.imgur.com/hJqhv6H.gif)
<br>
<br>


# References
Here some of the Books, Papers and Articles I read and Videos I watched in Preparation for my Project and Links to Websites which were very helpful in resolving issues and how-tos.

## Language, Libraries
https://www.python.org/<br>
https://jupyter.org/<br>
https://www.anaconda.com/<br>
https://www.tensorflow.org/<br>
https://keras.io/<br>
https://matplotlib.org/<br>
https://pandas.pydata.org/<br>
https://keras.io/api/preprocessing/image/<br>


## Sources for descriptions of Diseases
https://biomedical-engineering-online.biomedcentral.com<br>
https://www.kceyeclinic.com<br>
https://uvahealth.com<br>
https://www.asrs.org<br>
https://entokey.com/retina-4/<br>
https://www.atlasophthalmology.net/<br>


## Fundus Images with a mobile phone
Fundus Bilder sind Bilder der Netzhaut. Bislang wurden diese mit einer speziellen Kamera, der Fundus Kamera erstellt es wurden inzwischen jedoch auch Methoden entwickelt, um diese Bilder mit dem Handy machen zu können. Diese dann im selben Zug zu analysieren um eine Diagnose zu stellen, bedeutet einen großen Fortschritt in der Bekämpfung von Blindheit weltweit. 

https://journals.lww.com/ijo/Fulltext/2014/62090/Fundus_imaging_with_a_mobile_phone__A_review_of.16.aspx#:~:text=Fundus%20imaging%20with%20a%20fundus,coupled%20with%20a%20condensing%20lens.


## Transfer learning
Karen Simonyan, Andrew ZissermanVery Deep Convolutional Networks for Large-Scale Image Recognition<br>
https://arxiv.org/abs/1409.1556


## Image Preprocessing and Augmentation
Make a histogram of color channels<br> 
https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935<br>

OpenCV<br>
https://opencv.org/<br>

Pillow<br>
https://pillow.readthedocs.io/en/stable/handbook/tutorial.html<br>

DMENet: Diabetic Macular Edema diagnosis using Hierarchical Ensemble of CNNs
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220677

Deep Learning on Retina Images as Screening Tool for Diagnostic
Decision Support
https://arxiv.org/ftp/arxiv/papers/1807/1807.09232.pdf<br>
<br>

## Literature about AI in Healthcare

Deep Medicine by Eric Topol<br>
https://www.goodreads.com/book/show/40915762-deep-medicine

<br>

## Interesting Talks 
https://www.youtube.com/watch?v=ViSfhPE6q6Q&list=PL32IFKvW53Vg-gM2mq-tjF6edBqoTV_ie&index=5

https://www.youtube.com/watch?v=pMGLFlgqxuY&list=PL32IFKvW53Vg-gM2mq-tjF6edBqoTV_ie&index=3

https://www.youtube.com/watch?v=UZEstizNxkg&list=PL32IFKvW53Vg-gM2mq-tjF6edBqoTV_ie&index=1

<br>

## My Website
I have written about Data Science and worked on some projects, which can be found here:
http://patternrecognition.tech/
