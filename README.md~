## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, I then tried out my model on images of German traffic signs that I find on the web.


#**Traffic Sign Recognition**

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing number of images per class as well as one image per class .


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because color images will not add any extra information required for classification .
After that I normalize the image keeping the values between -1 to 1 so as to have 0 mean.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training , test and validation data was already present .
I did not add any augmented data .

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|--------------------------------------------------------------------------------|
| Layer         		|     Description	        					                         |
|:-----------------:|:----------------------------------------------------------:|
| Input         		| 32x32x3 RGB image   							                         |
| Inception 1x1     | Input: 32x32x3 Output: 32x32x32                            |
| RELU              |                                                            |
| Convolution 5x5,6 | 1x1 stride, valid padding, Input: 32x32x32 outputs 28x28x6 |
| RELU					    |												                                     |
| Max pooling	      | 2x2 stride, valid padding, Input: 28x28x6 outputs 14x14x6  |
| Inception 1x1     | Input: 14x14x6 Output: 14x14x14                            |
| RELU              |                                                            |
| Convolution 5x5,16| 1x1 stride, valid padding, Input:14x14x14 outputs 10x10x16 |
| RELU              |                                                            |
| Max pooling	      | 2x2 stride, valid padding, Input: 10x10x16 outputs 5x5x16  |
| Fully connected		| Input:5x5x16 , Output:400          									       |
| Fully connected   | Input:400 , Ouput:120                                      |
| RELU              |                                                            |
| DROPOUT           |                                                            |
| Fully connected		| Input:120 , Output:84          									           |
| RELU              |                                                            |
| DROPOUT           |                                                            |
| Fully connected		| Input:84 , Output:43          									           |
|-----------------: |:----------------------------------------------------------:|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used AdamOptimizer  as I was following LeNet Architecture .
For batch size I found that smaller batch size gave better results . I choose batch size of 150 .
Number or epochs - 30
I have used constant learning rate - 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.959
* test set accuracy of 0.934

I started with Lent without any modification ad gained an accuracy of 89 % .
Then I added dropout layer to avoid over fitting and was able to improve performance .
Finally I tried to add inception module but was facing some implementation issues due to problem with concat method.
So I thought of adding single inception layers in the network instead which improved validation accuracy .

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I tested on 11 images that I downloaded from the web .  Out of these 1 image of wild animal crossing was different from that in Germany . I used a sign with a kangaroo on it for experimental purpose . To my disappointment the network could not guess it correctly .The network needs to be improved .As far as other images , I have used images with different color combinations as well as night lighting which may make it difficult to classify.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

|:---------------------|:---------------------------------|:-------|
| Image			           |     Prediction	        					| Result |
|:-------------------- |:--------------------------------:|:------:|
| Keep Right           | Keep Right                       |Correct |
| 60 Km/h              | 80 Km/h                          |Wrong   |
| Yield					       | Yield											      |Correct |
| Go straight or left  | Go straight or left   					  |Correct |
| Priority Road   	   | Priority Road 										|Correct |
| Wild animals crossing| Roundabout Mandatory					    |Wrong   |
| 80 Km/h              | Yield                            |Wrong   |
| No vehicles          | No vehicles                      |Correct |
| 20 Km/h	             | 20 Km/h   							          |Correct |
| Ahead only           | Ahead Only                       |Correct |                
| No entry             | No entry                         |Correct |
|:---------------------|:---------------------------------|--------|


The model was able to correctly guess 8 of the 11 traffic signs, which gives an accuracy of 74%. This is lesser than  the accuracy on the test set .

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is a Keep Right (probability of 1), and the image does contain a keep right sign .


For the second image, the model is more sure that this is a Speed Limit (80 km/h) (probability of 0.95) and Speed Limit (60Km/hr) ( probability of 0.05), and the image is infact Speed Limit (60Km/hr). The prediction is wrong .

For the third image, the model is sure that this is a Yield (probability of 1), and the image does contain a Yield sign .

For the fourth image, the model is sure that this is a Go straight or left (probability of 1), and the image does contain a Go straight or left sign .

For the fifth image, the model is sure that this is a Priority (probability of 1), and the image does contain a Priority sign .

For the sixth image, the model is sure that this is a Roundabout Mandatory (probability of 1), and the image does not contain a roundabout mandatory sign,instead contains wild animal crossing sign.The prediction is wrong.

For the seventh image, the model is sure that this is a Yield (probability of 1), and the image does not contain a yield sign ,instead  contains Speed Limit (80 Km/hr) sign.The prediction is wrong.

For the eight image, the model is sure that this is a No vehicles (probability of 1), and the image does contain a No vehicles sign .

For the ninth image, the model is sure that this is a Speed Limit (20 km/hr) (probability of 1), and the image does contain a Speed Limit (20 Km/hr) sign .

For the tenth image, the model is sure that this is a Ahead Only (probability of 1), and the image does contain a Ahead Only sign .

For the eleventh image, the model is sure that this is a No entry (probability of 1), and the image does contain a No entry sign .
