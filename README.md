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

## Model

### Theory

Due to my current limited knowledge in time series, the first model I'll use and which was advised to put in place is convolutionnal networks and to be more specific Residual Convolutionnal network (ResNet).

I'll try multiple iterations of ResNet and even other convolutionnal based networks and put the best results of each ones in the Practice field.

When finished with iterations of convolutionnal network, I'll search for papers for ECG specific classification.
At this time, here's the papers I found and need to dig into:
- https://www.nature.com/articles/s41598-021-97118-5
- https://paperswithcode.com/task/ecg-classification
- https://www.frontiersin.org/articles/10.3389/fphy.2019.00103/full
