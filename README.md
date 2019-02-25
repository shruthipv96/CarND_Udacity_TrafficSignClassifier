# **Traffic Sign Recognition** 

## Writeup

---

**The project is to build a Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class_images]: ./document_images/class_images.png "class_images"
[class_hist]:   ./document_images/class_hist.png   "class_hist"
[preprocess]:   ./document_images/preprocess.png   "preprocess"
[optimizer]:    ./document_images/optimizer.png    "optimizer"
[web_images]:   ./document_images/web_images.png   "web_images"
[prediction]:   ./document_images/prediction.png   "prediction"
[soft1]:        ./document_images/soft1.png        "soft1"
[soft2]:        ./document_images/soft2.png        "soft2"
[soft3]:        ./document_images/soft3.png        "soft3"
[soft4]:        ./document_images/soft4.png        "soft4"
[soft5]:        ./document_images/soft5.png        "soft5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  The submission includes the project code.

Here is the [writeup](./Traffic_Sign_Classifier.md) for this project. The project notebook is [here](./Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**
* The number of samples in each class is as follows:

| Class       |     No. of images in each class	| Class       |     No. of images in each class	| 
|:-----------:|:-------------------------------:|:-----------:|:-------------------------------:| 
|  1 | 180 | 23| 330 |
|  2 | 1980| 24| 450 |
|  3 | 2010| 25| 240 |
|  4 | 1260| 26| 1350|
|  5 | 1770| 27| 540 |
|  6 | 1650| 28| 210 |
|  7 | 360 | 29| 480 |
|  8 | 1290| 30| 240 |
|  9 | 1260| 31| 390 |
|  10| 1320| 32| 690 |
|  11| 1800| 33| 210 |
|  12| 1170| 34| 599 |
|  13| 1890| 35| 360 |
|  14| 1920| 36| 1080|
|  15| 690 | 37| 330 |
|  16| 540 | 38| 180 |
|  17| 360 | 39| 1860|
|  18| 990 | 40| 270 |
|  19| 1080| 41| 300 |
|  20| 180 | 42| 210 |
|  21| 300 | 43| 210 |
|  22| 270 |
 
#### 2. Include an exploratory visualization of the dataset.

There is not so much information by understanding only the numbers. Let's visualize an image of each class along with the class names.

![alt text][class_images]

Let's visualize the number of images in each class.

![alt text][class_hist]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

Before training the network, I preprocessed the data in two steps:

1) **Grayscaling** : Convert the RGB image to gray scale image. This reduces the number of parameters and allows us to control the network more by reducing the dimensionality.
```python
    #Gray scale conversion
    gray = np.sum(image/3,axis=3,keepdims=True)
```

2) **Normalizing** : I normalized the gray scaled image so that it helps in converging the neural network faster. I used OpenCV NORM_MINMAX algorithm for this.
```python
    #Normalizing using cv2.NORM_MINMAX
    normalize = cv2.normalize(gray,None,0,255,cv2.NORM_MINMAX)
```

###### Find below the preprocessing at each step:

![alt text][preprocess]

You can see that the image became more clear and sharp after the preprocessing.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

When I used the LeNet neural network, I found that the validation accuracy did not go beyond 90% eventhough I tune the hyperparameters. So, I have modified and made the LeNet neural network architecture deeper so that the neural network will understand the data set and also improves the accuracy of the model.

My modified neural network has 14 layers of which:

* _Four_ are fully connected neural network
* _Three_ are convolutional neural network
* _Six_ Rectified Linear Unit (relu)
* _One_ max pooling


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 preprocessed image 				    | 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					| Outputs 28x28x6								|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU          		| Outputs 10x10x16								|
| Convolution 6x6		| 1x1 stride, VALID padding, outputs 5x5x20     |
| RELU					| Outputs 5x5x20 						        |
| Flatten			 	| Outputs 500									|
| Fully Connected		| Outputs 250									|
| RELU  			 	| Outputs 250									|
| Fully Connected		| Outputs 120									|
| RELU  			 	| Outputs 120									|
| Fully Connected		| Outputs 84									|
| RELU  			 	| Outputs 84									|
| Fully Connected		| Outputs 43 (No. of classes)					|
 
The code:

```python
def myNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    cv1_w = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean = mu, stddev = sigma))
    cv1_b = tf.Variable(tf.zeros(6))
    cv1   = tf.nn.conv2d(x,cv1_w,strides=[1,1,1,1],padding='VALID') + cv1_b

    #Activation.
    cv1_a = tf.nn.relu(cv1)
    
    #Pooling. Input = 28x28x6. Output = 14x14x6.
    cv1_p   = tf.nn.max_pool(cv1_a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    #Layer 2: Convolutional. Input = 14x14x6 Output = 10x10x16.
    cv2_w = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean = mu, stddev = sigma))
    cv2_b = tf.Variable(tf.zeros(16))
    cv2   = tf.nn.conv2d(cv1_p,cv2_w,strides=[1,1,1,1],padding='VALID') + cv2_b
    
    #Activation.
    cv2_a = tf.nn.relu(cv2)
    
    #Layer 3: Convolutional. Input = 10x10x16 Output = 5x5x20.
    cv3_w = tf.Variable(tf.truncated_normal(shape=(6,6,16,20),mean = mu, stddev = sigma))
    cv3_b = tf.Variable(tf.zeros(20))
    cv3   = tf.nn.conv2d(cv2_a,cv3_w,strides=[1,1,1,1],padding='VALID') + cv3_b
    
    #Activation.
    cv3_a = tf.nn.relu(cv3)

    #Flatten. Input = 5x5x20. Output = 500.
    cv3_f = flatten(cv3_a)
    
    #Layer 4: Fully Connected. Input = 500. Output = 250.
    cv4_w = tf.Variable(tf.truncated_normal(shape=(500,250),mean = mu, stddev = sigma))
    cv4_b = tf.Variable(tf.zeros(250))
    cv4   = tf.add(tf.matmul(cv3_f,cv4_w),cv4_b)
    
    #Activation.
    cv4_a = tf.nn.relu(cv4)
    
    #Layer 5: Fully Connected. Input = 250. Output = 120.
    cv5_w = tf.Variable(tf.truncated_normal(shape=(250,120),mean = mu, stddev = sigma))
    cv5_b = tf.Variable(tf.zeros(120))
    cv5   = tf.add(tf.matmul(cv4_a,cv5_w),cv5_b)
    
    #Activation.
    cv5_a = tf.nn.relu(cv5)

    #Layer 6: Fully Connected. Input = 120. Output = 84.
    cv6_w = tf.Variable(tf.truncated_normal(shape=(120,84),mean = mu, stddev = sigma))
    cv6_b = tf.Variable(tf.zeros(84))
    cv6   = tf.add(tf.matmul(cv5_a,cv6_w),cv6_b)
    
    #Activation.
    cv6_a = tf.nn.relu(cv6)

    #Layer 7: Fully Connected. Input = 84. Output = 43.
    cv7_w = tf.Variable(tf.truncated_normal(shape=(84,43),mean = mu, stddev = sigma))
    cv7_b = tf.Variable(tf.zeros(43))
    logits= tf.add(tf.matmul(cv6_a,cv7_w),cv7_b)
    
    return logits

```
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Before training the model, I set the hyper parameters for training the network which are:
```python
batch size   : 64      (smaller batch gave better results)
learning rate: 0.00075 (smaller rate to avoid false learning)
```
The EPOCH is not a fixed number. It varies in each training. I observed that the validation accuracy is getting saturated in the range of 93%-95% with this modified neural network. So, I did not find much use of training the model once it gets saturated. Hence, I am stopping the training once the validation accuracy goes beyond 93% for 10 times. This also avoids overfitting of the model.

**Note:** Though the EPOCH is not a fixed number, I don't want the training to happen infinitely. Hence, I limited to 50 times even if the validation accuracy does not goes beyond 93% for 10 times. But I have not yet faced this situation with this modified neural network.

```python
#Check if the validation accuracy crossed the limit for 10 times
if (validation_accuracy >= 0.93):
    count = count + 1
if count >= 10:
    count = 0
    break
```

Once the hyper parameters are fixed, optimizer should be chosen. The following graph shows the comparison of performance of few optimizers.

![alt text][optimizer]

From the study, I found that Adam Optimizer gives a better performance than others. Hence I chose the **Adam Optimizer**.

Once the hyper parameters and the optimizer is fixed, it's time to form the pipeline to train the data set.

* 1) Calculate the logits for the given data
* 2) Calculate the cross entropy of the softmax of the logits calculated
* 3) Calculate the loss of the cross entropy calculated
* 4) Using Adam optimizer, minimize the loss

The code:
```python
logits = myNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**My final model results were:**
* The notebook submitted has the training accuracy of 99.8% 
* The notebook submitted has the validation accuracy of 94.0%
* The notebook submitted has the test accuracy of 92.1%
* From the training and validation accuracy, I saw that the validation accuracy was 5.8% less than the training accuracy which means the model is overfitted . But still this can be accepted and the test accuracy is also 92.1% which is 11637 out of 12630 images are predicted correct.

**Achieving the solution:**
* I first tried training the Neural Network with the LeNet architecture because it is a very good classfier for signs like 0-9. Hence I chose that network for classifying the traffic signs.
* The architecture was able to classify the signs pretty good but the validation accuracy was around 88%
* So I reduced the batch size to 64 and learning rate to 0.00075 to improve the validation accuracy and I was able to improve the accuracy to 90% which is not sufficient.
* I felt that the LeNet architecture was deep enough to classify 10 signs but not as deep to classify 43 signs. Hence I added one more convolutional layer to enhance the network by understanding the features of the data set a little deep and one more fully connected layer to increase the depth.
* This modified network improved the accuracy as expected. I trained the model many times and I was able to see the validation accuracy in the range of 93% to 95% and predominantly in the range of 90%-92%. This helped me achieve the solution.

**Note:**
* I shuffled the training data before every training.
* I created small batches of the training data and trained them individually.
* Later I evaluated the accuracy of the training data and validation data.
* I repeated the training until the validation accuracy goes beyond 93% for 10 times.

The code for evaluating the model is available in the notebook in the `eleventh` code cell.

The code for training the network with the pipeline is available in the notebook in the `twelvth` code cell.

The code for evaluating the test data set is available in the notebook in the `thirteenth` code cell.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][web_images] 

The difficulties in predicting each image might be:

* **Test Image 1** : This image might get confused with general caution or pedestrians as they are also a black center line inside a triangle shape.

* **Test Image 2** : There are six classes of 2 digit speed limit {Speed limit (20km/h),Speed limit (30km/h),Speed limit (50km/h),Speed limit (60km/h),Speed limit (70km/h),Speed limit (80km/h)}. So I fear that the model might get confused between these classes. 

* **Test Image 3** : I do not find any difficulty with this image because it is somewhat unique in the shape. So there should not be any difficulty in classifying this image.

* **Test Image 4** : This image might get confused with Go Straight or Left class. But the probability is very less beacuse of the additional straight line in Go Straight or Left class.

* **Test Image 5** : Again, this image might get confused with Right-of-way at the next intersection or pedestrians as they are also a black center line inside a triangle shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction in visual form:
                                  
                                  Top 3 Prediction (First,Second,Third)

![alt text][prediction]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. As expected in the second test image, the top predictions are the 2 digit speed limit class. But because of proper training, the model was able to predict correctly.

But the test data set accuracy is 92.1% whereas here it is 100%. 

I suspect two reasons for this:

1) The web images covered only 5 classes of 43 classes. 100% accuracy in these classes gives me an understanding that these classes were also properly classified (most of them) in the test data set. The drop in accuracy of the test data set is mostly because of the rest 38 classes.

2) The accuracy reduces as the the size of data set increases.

The code for evaluating the web image data set is available in the notebook in the `fifteenth` and `sixteenth` code cells.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for reporting the top softmax probabilites is available in the notebook in the `seventeeth` code cell.

**The report of softmax probabilities of each test image....**

**Test Image 1:**

Top predictions and values:
* 1) Right-of-way at the next intersection[11] : 1.0
* 2) Double curve[21] : 1.08705e-25
* 3) Slippery road[23] : 3.84266e-29
* 4) Beware of ice/snow[30] : 2.70553e-29
* 5) Road narrows on the right[24] : 1.45119e-31
    
The difference of probability of the first prediction with second prediction is approximately 1. This implies that the model predicted the image to be `Right-of-way at the next intersection` with 100% probability. This shows that the model is very much certain in classifying the image. 

![alt text][soft1]

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Test Image 2:**

Top predictions and values:
* 1) Speed limit (30km/h)[1] : 1.0
* 2) Speed limit (50km/h)[2] : 2.68015e-08
* 3) Speed limit (70km/h)[4] : 3.90383e-12
* 4) Speed limit (80km/h)[5] : 3.63537e-15
* 5) Wild animals crossing[31] : 3.04342e-18
    
The difference of probability of the first prediction with second prediction is approximately 1. This implies that the model predicted the image to be `Speed limit (30km/h)` with 100% probability. This shows that the model is very much certain in classifying the image. 

![alt text][soft2]

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Test Image 3:**

Top predictions and values:
* 1) Priority road[12] : 1.0
* 2) Speed limit (20km/h)[0] : 0.0
* 3) Speed limit (30km/h)[1] : 0.0
* 4) Speed limit (50km/h)[2] : 0.0
* 5) Speed limit (60km/h)[3] : 0.0
    
The difference of probability of the first prediction with second prediction is 1. This implies that the model predicted the image to be `Priority road` with 100% probability. This shows that the model is very much certain in classifying the image. 

![alt text][soft3]

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Test Image 4:**

Top predictions and values:
* 1) Turn left ahead[34] : 1.0
* 2) Ahead only[35] : 1.50171e-12
* 3) Children crossing[28] : 3.75878e-20
* 4) Go straight or right[36] : 6.90799e-21
* 5) Bicycles crossing[29] : 6.86059e-21

The difference of probability of the first prediction with second prediction is approximately 1. This implies that the model predicted the image to be `Turn left ahead` with 100% probability. This shows that the model is very much certain in classifying the image. 

![alt text][soft4]

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**Test Image 5:**

Top predictions and values:
* 1) General caution[18] : 1.0
* 2) Turn left ahead[34] : 3.80724e-12
* 3) Traffic signals[26] : 1.60763e-13
* 4) Speed limit (60km/h)[3] : 1.31359e-18
* 5) Children crossing[28] : 5.24824e-19

The difference of probability of the first prediction with second prediction is approximately 1. This implies that the model predicted the image to be `General caution` with 100% probability. This shows that the model is very much certain in classifying the image. 

![alt text][soft5]

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

