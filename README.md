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
Two ways:
1. Download the binaries from [FreeByrd on SourceForge](https://sourceforge.net/projects/freebyrd/files/latest/download) or [FreeByrd on GitHub](https://www.github.com/jainris/FreeByrd/releases/latest/download/FreeByrd.zip)
    * Unzip the file
    * Run FreeByrd\\freebyrd.bat
1. Build this repository.
    * Requires Python (only tested on 3.7 64-bit)
    * Requires Flutter

---
## Note
* Made with Windows as platform in mind.
* Made with help from:

|GitHub Username|Helped with/Contributions|
|---------------|-----------------------|
|@rmlshao|[Python Output Gen](./ML/output.py), [Utils](./ML/Utils)|
|@max-kaufmann||
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
