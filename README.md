# EventSegmentation

Contains the SEM model and a few basic demonstrations. Very much a work in progress and the documentation here lags as 
a consequence.

The main code is listed in the `models` module:
* `models.sem`: contains the code for the SEM model
* `models.event_models`: contains code for the various neural network models used by SEM. They all 
    share a similar structures
    
There are a few prepackaged demonstrations in Jupyter notebooks. These have been pre-run and can be opened on github
without installation:
* `Demo - Toy Data (Segmentation)`: These simulations demonstrate how SEM can segement simple, 2D dynamical systems with
various different methods of estimating the event dynamics of the system.
* `Demo - Structure`: These simulations show how SEM draws event boundaries that respect structure in vector space (from the paper)
* `Demo - HRR.ipynb`: Demonstration of the Holographic reduced representation
* `Demo - Toy Data.ipynb`: Simulations of the SEM model segmenting 2D data sets
* `Demo - Motion Capture Data.ipynb`: Simulations of the SEM model on the 3D motion capture data. 


#### Installation Instructions

This library run on Python 2.7 and uses the tensorflow and keras and libraries for neural networks. 

I recommend using Anaconda python and a virtual environment. [You can find instructions to install Anaconda
 here](https://docs.anaconda.com/anaconda/install/).

Once you have anaconda installed, you can install a virtual environment by running

    conda env create --file environment.yml

This will install everything you need to run the demonstration notebooks.

You'll need to activate the virtual environments and open jupyter to access the demonstration notebooks. To do so, run

    source activate sem
    jupyter notebook


To deactivate the virtual environment, run

    source deactivate


##### Known issues:

* Progress bars seem to have problems with this installation.