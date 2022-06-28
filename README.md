# ABSTRACT


Pneumonia is an infection which afflicts the pulmonary system of the body it manifests in. It causes the air sacs (alveoli) of the lungs to fill with fluid or pus, which in turn causes cough with phlegm or pus, fever, chills and labored breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia. Bacteria such as Streptococcus pneumonia and Mycoplasma pneumonia are known to cause mild cases of pneumonia. COVID-19, the disease the novel coronavirus causes, can spread to the lungs, causing pneumonia. While many people recover, some develop severe pneumonia that does not respond well to treatment. Doctors can differentiate between a normal and a pneumonia afflicted patient by analyzing their chest X-rays. Doctors can also distinguish between viral and bacterial pneumonia by examining the chest X-ray of the patient. And If the disease is identified in a more user friendly way. The treatment can be taken care of by medical industries.

# 1. Introduction 


* 1.1                 Objective and goal of the project

The projects objective is to design a Convolution Neural Network and train it by using Machine Learning algorithms to detect and differentiate between normal and Pneumonia afflicted patients by analyzing their Chest X-Rays. Additionally a well defined UI will be designed on the front end to create a user friendly interface, where all the popups, messages and information about the user are informed. The system will also be aimed at providing security in data. 
* 1.2                 Problem Statement 

Pneumonia detection using chest X-rays has been an open problem for way too long. Many machine learning models have been extensively studied by the researchers  and a lot of advancements have been brought from the day the research started. One of the papers segmented the lung regions from chest X-ray images and extracted eight statistical characteristics. The paper also implemented 5 ML classifiers: multi-layer perceptron (MLP), random forest, sequential minimal optimization (SMO), classification via regression, and logistic regression. But the problem with the Machine Learning algorithms is, the process involves handcrafted features that need to be extracted and selected for classification or segmentation. Additionally most of the solutions found so far have used almost every Machine Learning model combination possible. The project wanted to take an alternative approach rather than Machine learning models. Also, none of the existing projects so far have a proper UI interface and the data security since it involves the data of an individual.








* 1.3	          Motivation


Pneumonia affects a large number of individuals, especially children, mostly in developing and underdeveloped countries characterized by risk factors coupled with the unavailability of appropriate medical facilities. Early diagnosis of pneumonia is crucial to cure the disease completely. Examination of X-ray scans is the most common means of diagnosis, but it depends on the interpretative ability of the radiologist. Thus, an automatic CAD system with generalizing capability is required to diagnose the disease.

* 1.4	          Challenges


There were quite a few challenges when the work for this project started. The whole project was split into three divisions within ourselves. Front End, Back end and Machine Learning Model. Every phase of the project working had its own problems of their own. Since the data deals with information of patients' health conditions it has to be properly secured. For which we have to make the website trustworthy to the user. Secondly, the ML model. The project used an existing dataset from Kaggle, on which preprocessing is performed. The process where we created two arrays one with the images and other with the labels respective to images(o if no pneumonia & 1 if has pneumonia). The major problem was covering the model. When we started fitting the model, since the dataset is of 2.5 GB and the platform used was Google collab. It frequently kept on crashing before implementing the model. Because of which only 10 epochs could be implemented however we further tried to implement and covnerge the algorithm. Even the backend was secured properly.

# 4 System Design

CNN-based deep learning algorithms have become the go-to resource for medical image analysts. Medical image classification plays an essential role in clinical treatment and teaching tasks. The deep neural network is an emerging machine learning method that has proven its potential for different classification tasks. Convolutional Neural Network dominates with the best results on varying image classification tasks. We have applied the convolutional neural network (CNN) based algorithm on a chest X-ray dataset to classify pneumonia. The algorithm used to train the Neural Network is called Transfer Learning algorithm. Transfer Learning is a simple and popular approach in deep learning where a model developed for a task is reused as the starting point for a model on a second task. We have also intended to prime our input dataset by de-noising it with the help of filters. Different numbers of epochs and filters have been used in the CNN to ensure the highest possible output accuracy. The output displayed via a Confusion Matrix is to showcase the accuracy and precision of the model designed.




#5Implementation of System
![image](https://user-images.githubusercontent.com/61506157/176119783-55b64201-d605-4549-873b-3450a874f8fd.png)



*5.1 DOCTORS METHODOLOGY: 

            The project’s primary doctor will begin by asking the user about their medical history and symptoms. User will also undergo a physical exam, so that your doctor can listen to your lungs. In checking for pneumonia, the doctor will listen for abnormal sounds like crackling, rumbling or wheezing. If the doctor thinks user might have pneumonia, an imaging test may be performed to confirm the diagnosis. One or more of the following tests may be ordered to evaluate for pneumonia: Chest x-ray: An x-ray exam will allow the doctor to see user’s lungs, heart and blood vessels to help determine if user has pneumonia. When interpreting the x-ray, the radiologist will look for white spots in the lungs (called infiltrates) that identify an infection. This exam will also help determine if user has any complications related to pneumonia such as abscesses or pleural effusions (fluid surrounding the lungs).

               

*5.2 ALGORITHM USED :

5.2.1 Dataset Preprocessing:
A function to label the images in the dataset based upon whether the person is normal or diagnosed with pneumonia. Normal patients have been labeled 0 Pneumonia diagnosed patients have been labeled 1. This function takes the dataset directory as an argument written to perform pre-processing on the image dataset which includes Image gray-scaling, Image resizing, Storing the grayscale values and labels in x and y variables. The function is essentially used to assign labels to the images depending on whether they are normal x rays or pneumonia x-rays, resize the image and store the images in the array X and the labels in the array y
 
![image](https://user-images.githubusercontent.com/61506157/176119329-21be4dcc-3dc0-4e1c-9d9f-5014eb9a8bf1.png)

5.2.2 Training and Testing data:
The wohle code is designed in such a way that it prints the shape of X_train and X_test. The output (5216, 150, 150, 3) means that X_train has 5216 images which are 150x150 pixels and they are colored images (not grayscale). X_test has 624 images which are 150x150 pixels and they are colored images (not grayscale). Similarly its also prints the shape of y_train and y_test. They have two columns; one column is pneumonia or normal and the other column has values of 1 or 0
![image](https://user-images.githubusercontent.com/61506157/176119368-fee0b81e-b375-461a-a402-e3488f74ed5b.png)

 


4.2.3 Pneumonia images function:
Define a function to plot images of x-rays that are normal and those that are of pneumonia. Using matplotlib to demonstrate pneumonia and no pneumonia images side by side. Models often benefit from reducing the learning rate once learning stagnates. For this, we used ReduceLROnPlateau which monitors accuracy setting the factor by which to reduce learning rate as 0.1. Verbose is 1: update messages. 
![image](https://user-images.githubusercontent.com/61506157/176119379-d83f180f-0ae9-499c-b9cc-88793cd2120e.png)


 
 ![image](https://user-images.githubusercontent.com/61506157/176119408-bf0f2095-c81f-4849-abca-0b144843426a.png)
![image](https://user-images.githubusercontent.com/61506157/176119419-56853822-4d5f-4cd3-b370-6dce2014857b.png)

 



5.2.4 Transfer Learning Algorithm: 

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems. We aim to apply the convolutional neural network (CNN) based algorithm on a chest X-ray dataset to classify pneumonia. The algorithm used to train the Neural Network is called the Transfer Learning algorithm. Transfer Learning is a simple and popular approach in deep learning where a model developed for a task is reused as the starting point for a model on a second task. We also intend to prime our input dataset by de-noising it with the help of filters. Different number of epochs and filters will be used in the CNN to ensure the highest possible output accuracy. The output will be displayed via a Confusion Matrix to showcase the accuracy and precision of the model designed.
 
 
![image](https://user-images.githubusercontent.com/61506157/176119483-4d4eccd1-684e-421d-bb25-4588c4f485ba.png)
![image](https://user-images.githubusercontent.com/61506157/176119489-f8c4f319-4a20-4eee-8dbe-e685707184c1.png)









5.3 BACKEND SERVER:
	5.3.1 Creating server for front-end
Using Flask Technology, a set up is made with local virtual environment which is used to develop and deploy a flask server to host the front-end web application on which backend was also integrated with the machine learning model. 
 
![image](https://user-images.githubusercontent.com/61506157/176119505-b271485f-587f-4c8e-99fe-38410c9b824b.png)

	5.3.2 Encrypting and storing data
After the user inputs their data onto the front end, the sensitive information received on the flask backend like their name, email, location, etc are encrypted using MD5 encryption before storing them onto the MongoDB database.
The sensitive encrypted data along with their pneumonia detection result from the “.h5” ML model is stored in the NoSql MongoDB database.
 ![image](https://user-images.githubusercontent.com/61506157/176119523-bec4e8cf-41b7-4e12-a9f7-5bec61014ac0.png)

	 
	![image](https://user-images.githubusercontent.com/61506157/176119535-ba9bd1a7-b011-4a23-bd65-3ceae6ce5dd1.png)

 
5.3.3 Integrating Machine Learning Model

After training the ML model on google collab, the ML model was downloaded as a “.h5” file to integrate into the flask server which was then used for predicting the presence of pneumonia when an user uploads the X-Ray scan on the website. Once the image is uploaded by the user, the image is preprocessed by converting it into grayscale,  resizing and reshaping which helps in better prediction.
        


![image](https://user-images.githubusercontent.com/61506157/176119550-4d6e1a27-26a3-4732-a6da-69a28829098f.png)

# 6. Results and Discussion

Detection of Pneumonia by Analyzing Chest X-Rays  is a small step towards revolutionizing the application of image processing using Machine Learning and Neural Networks in the field of medicine. Our model was able to deliver an accuracy of 97%  for detecting pneumonia. Transfer Learning Algorithm was the core of this project which helped in getting  highly accurate data at great speeds. Our goal in mind while coming up with an idea for this project was to ease the burden shouldered by millions of doctors across the world who are working ceaselessly to treat people from the devastating effect of COVID-19 caused by the novel coronavirus. Since a COVID-19 afflicted patient displays symptoms of pneumonia, a disease which can also be contracted through bacterial means, it becomes increasingly difficult to differentiate between the two types of patients based on a human's perspective on their chest X-rays and to treat the patients with the proper care that they need. There have been several instances of mislabelling patients with similar pneumonia-like symptoms due to the sheer volume of COVID-19 afflicted patients and this hampers the treatment. Our project could be a first step towards making the process of identification and differentiation of pneumonia through a machine’s analysis of chest X-rays. By off-loading some of their work onto machines, doctors can spend more time attending to more pressing matters and can also safely and reliably treat their patients.








