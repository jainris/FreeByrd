# FreeByrd
---
## Introduction
This application is made to help RSPB in monitoring bird colony population size, by taking long audio files and returning timestamps where bird call is detected. The bird calls are also grouped together on basis on similarity to form clusters. The number of seconds and a sample audio of each cluster is also returned. The user can use this information to identify the species of a cluster and thus know the number of seconds of bird calls recorded for each species, thus enabling them to gauge the population size for that species.

---
## Implementation
Our solution implements a pipeline, which consists of 3 parts:
1. Segmentation
    * Uses birdvoxdetect (an ML Model for segmentation) with some additional noise cancellation and post-processing to generate time ranges with bird calls within them. 
1. Feature Extraction
    * Uses birdNET (an ML Model for classification) to generate feature vectors.
1. Clustering
    * Two Strategies implemented:
        1. Uses Dominant Set Clustering.
        1. Uses a Hybrid of SOM and K-Means Clustering.

---
## Installation
1. The folder named [Release](./Release) contains the [main application executable](./Release/freebyrd.exe).
1. Requires Python (only tested on 3.7 64-bit)
    * Download Link: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe
    * Guide: https://wiki.python.org/moin/BeginnersGuide/Download
    * The Path variable must be set to include python and it's libraries
     (refer to https://docs.python.org/3/using/windows.html for how-to)

---
## Note
* Made with Windows as platform in mind.
* Made with help from:

|GitHub Username|Helped with/Contributions|
|---------------|-----------------------|
|@rmlshao|[Python Output Gen](./ML/output.py), [Utils](./ML/Utils), [birdNET Preprocessing](./ML/FeatureExtraction/birdNET_preprocess.py) |
|@max-kaufmann|[birdNET adaptation](./ML/FeatureExtraction)|
|@Alex4877||
|@erin-cox||

---
## Licenses
#### Dominant Set Clustering
Dominant Set Clustering is adapted from https://github.com/feliksh/SCDS
and so uses GNU GPL v3.0 license, which has been included in its directory
([ML\Clusterer\LICENSE](./ML/Clusterer/LICENSE))
#### BirdNET
Feature Extraction uses BirdNET model https://github.com/kahst/BirdNET
and so uses MIT license, which has been included in its directory
([ML\FeatureExtraction\LICENSE](./ML/FeatureExtraction/LICENSE))
#### Rest
Rest of the project is under [MIT License](./LICENSE)