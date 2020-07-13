# **Self Driving Car ND** 

## Deep Learning

### Traffic Sign recognition

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: count.jpg "Train Visualization"
[image2]: count1.png "Valid Visualization"
[image3]: count2.png "Test Visualization"
[image4]: ./Mysigns/8x.png "Attention"
[image5]: ./Mysigns/2x.png "speed limit 30"
[image6]: ./Mysigns/3x.png "priority road"
[image7]: ./Mysigns/road_works.png "road works"
[image8]: ./Mysigns/traffic_light.png "Traffic light"


## Rubric Points
Submission Files -> Done
Dataset Summary -> Done
Exploration Visualization -> Done
Preprocessing -> Done
Model Architecture -> Done
Model Training -> Done
Solution Approach -> Done
Acquire New images -> Done
Performance Analysis -> Done
Model certainty -> Done





---
## Report


### Data Set Summary & Exploration
#### Loading Datasets
A pickle file containing Train,Valid,Test Data.I used pickle library to load various data into their respective array variables.If there was a single data repo, I would have used SKlearn's train_test_split() with 80% of Data alloted to train and 20% to split.After loading data, We have three sets X_train,y_train,X_test,y_test,X_valid,y_valid.You might wonder whats the use of test and valid data sept split . While validitation even though doesnt take part in model training, but only gives an evaluation of model.Its information can actually leak into the training process when we consistently change hyper parameter such that both train and validation accuracy are high with less the 8% difference between them.Thus to receieve an accurate estimate of how a mdel perfoms, we must test it with data it hasnt seen . Also in the Data set X stands for features(Input) and Y stands for labels these features represent(Output).

I used numpy to get summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

A bar count plot of each train,test and valid set was plotted to understand the distribution of data.
One good thing about the dataset is that the distribution of signs are remarkably similar in distribution across the three data sets .This means test and valid are good valid sets for model evaluation as they can say how well the model performance for this use case. However there is also little bias in the distribution in terms of counts for first 30% fo sign type , which means it might not be able to predict all signs equally with good accuracy.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Data Preprocessing

As a first step, I decided to convert the images to grayscale because I was about to build a model inspired from LeNet to solve my classification problem.According to the architecture ,It takes (None,32,32,1) shape tensor as input.Which means it expects a single channel image.However we have 3 channel image(RGB).There was 2 ways to handle this issue.
1. Add a convolution layer before the input of the network such that it convolves the (32,32,3) tensor into a (32,32,1)
2. Convert the RGB image into single channel grey scale.

At first,I thought solution 1 would be better to solve the channel incompatabilties as it would hold information regarding 3 channels in better way.But in practise, I found out greyscale conversion had better train performance for the network.However there can be some other architecture that can use the first one to have much better performance.But in our case going for second method as it satisfies our usage.

In next stage the image was normalised using its mean . This process has the nature of changing our search space into something more symmetric and eventually contributing to convergence performance.One thing , I ntoiced is that if we are going for first method of solving channel incompatabilities .Its better to avoid normalisation as it seems to penalise the performance of the network.

I did ponder on generating more dataset as it will always be a definite improvement to network training. However over course of training the network, it was found given data was more than enough to satisy the need.If at all, I decided to go for more data rather prefer real world data than augumented data.

 


#### 2. Neural Network Model
I studied the Lenet and implemented it for our use case however the performance was not satisfactory thus changed the model.
I used a consistent 3x3 convolution to preserve as much information as possible and sub sample it using Max Pooling layer.In the end used 5 Fully connected layers to more accurately learn classifcation.
My final model consisted of the following layers:

| Layer            | Description                                                 |
|------------------|-------------------------------------------------------------|
| Input            | 32x32x1 grey image  and zero mean normalised                |
| Convolution      | 3x3 filter with 1x1 stride, valid padding, outputs 30x30x6  |
| RELU             |                                                             |
| Max Pooling      | 7x7 Kernel with 1x1 stride, valid padding, outputs 24x24x6  |
| Convolution      | 3x3 filter with 1x1 stride, valid padding, outputs 22x22x16 |
| RELU             |                                                             |
| Max Pooling      | 6x6 kernel with 1x1 stride, valid padding, outputs 17x17x16 |
| Convolution      | 3x3 filter with 1x1 stride, valid padding, outputs 15x15x32 |
| Flatten          | outputs 7200                                                |
| Fully Connected  | outputs 400                                                 |
| RELU             |                                                             |
| Dropout          | keep_prob=Dynamic                                           |
| Fully Connected  | outputs 200                                                 |
| RELU             |                                                             |
| Dropout          | keep_prob=Dynamic                                           |
| Fully Connected  | outputs 120                                                 |
| RELU             |                                                             |
| Dropout          | keep_prob=Dynamic                                           |
| Fully Connected  | outputs 84                                                  |
| RELU             |                                                             |
| Fully Connected  | outputs 43                                                  |
| Softmax          |                                                             |
|                  |                                                             |
|                  |                                                             |

 


#### 3. Model Training

Batch size : 128
Epochs :80
learning_rate : 0.00097
Optimizer : AdamOptimizer
Loss : Softmax_crossentropy
Drop_out :1,0.2,0.5,0.7

#### 4. Approach and results

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 93% 
* test set accuracy of 91%

So first,I began with LeNet with simple modification.The inpu layer was convolution layer that changed (32,32,3) to (32,32,1) . The model overfitted fast with validation accuracy hovering at 40% and train clode to 90% , Also it took lot of epochs reach this accuracy level.On contrary when i used grey scale conversion validation accuracy increased by 20-23% with similar levels of overfitting. Another difficulty , I faced is the learn rate goldilock zone was very narrow 0.001 to 0.0009. I though the network was not having enough parameters to generalize enough added 2 more layers of fully connected layer.It didnt improve the model .Thats when i changed weights initalisation to gloriot() or xavier initialisation.With this the model had immediate jump to 84% of validation accuracy.I finally zeroed in on above model but with dropout only in first connected layer . I kept changing dropout rate evelntually pushing the model to 88% , but problem was the drop out was around 0.7 . Which means, I am loosing lot of information and even though model was good in validation . I was wary that it wont perform as such from internet downloaded images . Thus, I adopted above model with dropout rate of 0.7 the validation accuracy crossed 93 % but only after 61 epochs . So I adopted a dynamic dropout approch depending upon overfitting status , I changed fropout rate in training by 1,0.2,05,0.7.Using this model achieved 93% convergence in 42 epochs.

Things to do in future.
I am submitting the project as base objective of the project is achieved.However I want to try out few things.

1.Still going to try the intial (32,32,3)->(32,32,1) input convolution layer as it has to hold better representation of information held by the image than greyscale.During my experimentation , I didnt try gloriot initalisation with this input approach. Thus its my belief that along with gloriot initalisation , l2 regularistaion and batch normalisation ,its possible to build a far more superior network than described above.

2.The currently presented network also can have L2 regularisation and Batch normalisation to increase performance more than current 93% validation accuracy.
 

### Test a Model on New Images

#### 1.Test images choosen from internet
I have downloaded test data from 2 different distribution for german traffic signs.The distributions are of Photographic images similar to  train data and cartronized logo further from train data. My guess is even though the network is trained on photographic dataset , if it has learnt good enough features then there will atleast some correct predictions of cartoon data.

Here are some German traffic signs that I found on the web:


![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



#### 2.Model Predictions

Here are the results of the prediction:

| Image                | Prediction           |
|----------------------|----------------------|
| Keep right           | Keep right           |
| wildlife crossign(c) | Road work            |
| priority road        | Priority road        |
| traffic lights(c)    | Traffic signals      |
| road works(c)        | Road work            |
| Bicycle Only(c)      | Priority road        |
| Railway crossing(c)  | Bumpy road           |
| Attention            | General caution      |
| speed limit 30       | Speed limit (30km/h) |
| right of way         | Right-of-way         |


The model predicted 7/10 with estimated 70% accuracy . 2/3 cartoonised images sucessfully classfied.
On contrary the test set gave a 91% accuracy . Which suggest the model is performing well for given data distribution, at the same time learning features that allow it to detect some cartoonized images as well.

#### 3. Soft_max Predictions.

The softmax predictions are displayed in last-1 cell.
How to interpret the data ?
lets take first example.
first line gives the network prediction priority road
next we have list of top 5 softmax probabilites.
Then we have list of sign names.These sign names correspond to above probabilities in the same order.


Now lets check 3 cases where the model failed . This will actually give us an idea on how far the model was off from the ground truth.

Case I

| Image                | Prediction                                           |
|----------------------|------------------------------------------------------|
| wildlife crossign(c) | Road work (1.00000000e+00)                           |
|                      | Dangerous curve to the left(2.57462816e-08)          |
|                      | Dangerous curve to the right(2.61877908e-10)         |
|                      | Right-of-way at the next intersection(1.96025557e-11)|
|                      |  Beware of ice/snow(3.54929971e-15)                  |

The prediction was way off and not even close even if we take image morphology into consideration.
We must note that this image is from a different distribution that was not trained on the network.
This was a cartoonised logo . Fact that wildlife crossing sign count in our train data was also generally low.
All these odds stacked on our network made it fail in classifying this sign.



Case II

| Image                | Prediction                                           |
|----------------------|------------------------------------------------------|
| Bicycle Only(c)      | Priority road (0.83650011)                           |
|                      | Right-of-way at the next intersection(0.15168469)    |
|                      | Beware of ice/snow(0.00449141)                       |
|                      | Dangerous curve to the left(0.00190808)              |
|                      | Vehicles over 3.5 metric tons prohibited(0.00183043) |

Case II is same as CASE I . The model failed to classify it dues to similar lack of data.


Case II

| Image                | Prediction                                           |
|----------------------|------------------------------------------------------|
| Railway crossing(c)  | Bumpy road (1.00000000e+00)                          |
|                      | Traffic signals(1.97972572e-08)                      |
|                      | Bicycles crossing(6.99574816e-12)                    |
|                      |  General caution(2.73562683e-12)                     |
|                      | Road work(2.11828775e-12)                            |

Case III is interesting one . This class of sign was never trained on network.I was curious to find one sign board will it find similar. It classify it to be bumpy road with maximum probability. Interesting ly the mountain edges seem to be similar to the upper portion of X in railway crossing. This made the classifier to classify it as bumpy road ?

















