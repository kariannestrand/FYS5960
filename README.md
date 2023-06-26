# FYS5960  
# The work for my master's thesis "Using machine learning to predict the effect of Deep Brain Stimulation on axonal pathways".  

The **[dbs.py](https://github.com/kariannestrand/FYS5960/blob/main/dbs.py)** file contains two classes that form our model, simulating the interactions between stimuli and axons. The `Axon` class shapes and constructs the cell instance using LFPy 2.0 and the NEURON simulator. The `Simulate` class generates various stimulus and axon parameters, and initiates the NEURON simulations.  

Furthermore, the file **[generate.py](https://github.com/kariannestrand/FYS5960/blob/main/generate.py)** includes two classes. The `DataFrame` class calculates the minimum distance from the stimulus to the axon, declears the detection of a spike, and generates dataframes with the desired configuration. The `Visualizer` class generates a 3D plot and a subplot to provide an overview of the simulations.  

Finally, the entire machine learning algorithm is found in the **[ml.py](https://github.com/kariannestrand/FYS5960/blob/main/ml.py)** file, where **PyTorch** is used to create a three layered neural network. Related plots are also generated within the same file.  


The **[run_sim.py](https://github.com/kariannestrand/FYS5960/blob/main/run_sim.py)** file is used to execute the classes within the files **dbs.py** and **generate.py**. You can modify the variables within **run_sim.py** as needed. To execute the simulations, simply write the following command in the command line:  

    Python3 run_sim.py  

Depending on the command line inputs provided, this will perform the simulations and generate dataframes with 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, and 200 points along the axon as parameters, resulting in a total of 13 dataframes. These dataframes will be saved in the **[Data](https://github.com/kariannestrand/FYS5960/tree/main/Data)** folder. As an example, we have created dataframes with a sample size of 1000 for both straight and curved axon simulations.  

The **[run_ml.py](https://github.com/kariannestrand/FYS5960/blob/main/run_ml.py)** file is used to execute the classes within the **ml.py** file. The variables within **run_ml.py** can be modified as desired. To initiate the training and obtain the performance results of the machine learning model, simply write the following command in the command line:  

    Python3 run_ml.py   


