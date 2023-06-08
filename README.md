# ECG_Myocardial_Infarction

This project is created because I wanted to experiment on time-series related data and in the healthcare division.
After checking for some healthcare related dataset on http://timeseriesclassification.com/
The dataset found is corresponding to both my needs.

## Dataset

The dataset used is from http://timeseriesclassification.com/description.php?Dataset=ECG200
Which is extracted from the Thesis "Generalized feature extraction for structural pattern recognition in time-series data" of R. Olszewski (https://dl.acm.org/doi/book/10.5555/935627)

## Goals

The goals for this project is to:
- Understand the data
- Rightly  manipulate data
- Find a good model that fit the datatype
- Approach the 89% as much as possible

## Data & Context

To better understand how to classify the data, I need to learn to do it myself, prior to make a NN learn it.
This way, here's a list of sources/links I've found/used to help me.

For ECG/EKG comprehension articles/web:
- https://www.osmosis.org/learn/ECG_cardiac_infarction_and_ischemia
- https://www.mayoclinic.org/diseases-conditions/myocardial-ischemia/symptoms-causes/syc-20375417
- https://jamanetwork.com/journals/jamainternalmedicine/article-abstract/600227
- https://oxfordmedicaleducation.com/ecgs/ecg-interpretation/
- https://en.wikipedia.org/wiki/Myocardial_infarction
For ECG/EKG videos:
- https://www.youtube.com/watch?v=CNN30YHsJw0
- https://www.youtube.com/watch?v=TTYGxK1DNKA

With this knowledge I'm able to detect what's wrong with the given ECG

### Context

The dataset used is the result of the subject of the thesis.
From the preview of the thesis paper, it doesn't specify if the heartbeats are extracted from a specific EKG/ECG lead (I,II,III,V1-V6,aVR,aVL,aVF).
This information could've been useful to determine some example of the specific lead to not find suspicious data (like inverted QRS complex for ex).
This isn't bad but could've been helpful for further "investigation" or tests.

## Basic Model

### Theory

Due to my current limited knowledge in time series, the first model I'll use and which was advised to put in place is convolutionnal networks and to be more specific Residual Convolutionnal network (ResNet).

I'll try multiple iterations of ResNet and even other convolutionnal based networks and put the best results of each ones in the Practice field.

When finished with iterations of convolutionnal network, I'll search for papers for ECG specific classification.
At this time, here's the papers I found and need to dig into:
- https://www.nature.com/articles/s41598-021-97118-5
- https://paperswithcode.com/task/ecg-classification
- https://www.frontiersin.org/articles/10.3389/fphy.2019.00103/full

### ResNet

Since ResNet is mostly used for image classification it's algorithm require 2D Conv blocks which translate to (N,Cin,H,W) array in pytorch.
The option to change the first layer to a 1D Conv Block can be helpful but need to rebuild the links to 2d Conv afterwards, In this case we should need to rebuild entire Resnet with 1D Conv blocks.
I've chose to "expand" my arrays to the required 4D array required by Pytorch without changing the necessary informations.
I don't want to spend many time tuning a specific ResNet model for a good accuracy, If I've brokent biad / variance, I'll rather spend this time to study a better model best suited for time series and/or for ECG.

In Addition, I've read this paper which is a great reminder for how ResNet should work: https://arxiv.org/pdf/1611.06455.pdf

For quicker processing, the batchsize is set at 50
Using Adam Optimizer with 0.001 of LR
Train Dev repartition is 60/40
Using the simple Accuracy metric

#### No scheduler

After running without any scheduler for 500 & 5000 Epochs for only train & dev, here's the dev accuracies registered:
![Image](LossGraphRN_NoScheduler.png)
![Image](AccuracyGraphRN_NoScheduler.png)

The maximum found is around 80% Accuracy

##### With Scheduler

I've chosen the ConstantLR Scheduler, after some quick tests (500 epochs each), the best value found for it is 0.2 and effective every 5 epochs.
After all the tests done, this helped to increase global accuracy per 10%.
This is the RSSched.pth in the weight folder.

After full processing (train/dev/test), test accuracy is up to 84%.

![Image](LossGraphRN_SchedON.png)
![Image](AccGraphRN_SchedON.png)

Still need to check for false positive/negative.


## Other Model tested

Here's the other models I'll test and/or try to implement for the ECG:
- LSTM

There'll be more difficult model when using the ECG5000 Dataset, for this one, keeping it simple. 

### LSTM

To better understand and getting experience on LSTM, using Kaggle's Competition at https://github.com/antoninchafiol/Kgl_TS_StoreSales

