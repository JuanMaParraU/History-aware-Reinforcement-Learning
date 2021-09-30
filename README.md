# History-aware-Reinforcement-Learning
## ABS SAS

Mobile communications case study using autonomous [airborne base stations](https://ieeexplore.ieee.org/abstract/document/9448192). The system uses RL for positioning the simulated drones while maximising the covered end users.

##User Guide for linux/osx command line 

1. Check python version: 
```python 
python --version  
```
2. Download anaconda (check python 2 or 3 and select the appropriate Anaconda 2 or 3): 
```python 
cd /tmp 

curl -O https://repo.anaconda.com/archive/AnacondaX-2019.03-Linux-x86_64.sh 

bash AnacondaX-2019.03-Linux-x86_64.sh 
```
3. Check installation:   
```python
 conda -h
```
 
4. Create and activate anaconda environment (“drones_env”) the environment should be created with Python 3.X as interpreter: 
```python
conda create --name drones_env python=3.X 

conda activate drones_env 
```
5. Install numpy 
```python 
conda install -c anaconda numpy 
```
6. Install pandas 
```python
conda install -c anaconda pandas 
```
7. Install torch 
```python
conda install -c pytorch pytorch 
```
8. Install matplotlib 
```python 
conda install -c conda-forge matplotlib 
```
9. Install paho-mqtt. Check docmuentation about MQTT [here](https://www.eclipse.org/paho/index.php?page=clients/python/docs/index.php)
```python 
conda install -c conda-forge paho-mqtt
```
10. Run script 
```python 
python main.py
````
