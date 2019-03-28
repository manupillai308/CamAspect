# CamAspect
A Smart Security Surveillance System
  ## A centralized system for human identification and tracking.
Person detection and tracking is one of the most important research field that has gained a lot of attention in recent years. There are lot of  surveillance cctv cameras installed around us thus, it is necessary to develop a computer vision based technology that automatically processes the real time video frames in order to track the person.
Here, we aim to build an intelligent system which has its own secure database system, a cctv network and a network of connected nodes(systems). Each node will be connected to the centralized system. Node will be used by authorized personnel with their login credentials for their authorized operations in the system(searching,tagging,etc)

  ## Objectives:
  #### 1. Developing a centralised mainframe
  It will be installed on a single centralised machine where the main algorithm will be running and the database is
  accessed. This will be connected to the nodes from which the authorized personnel will get access to the system.
  #### 2. Algorithm-Human identification and tracking system
The algorithm will detect humans in the cctv footage frames and will search the database for identification of people in the live video footage. If there is one person of interest in the live feed, there will be an option to select that individual and only track activities of that person. After tracking, the algorithm will upload the logs of tracking to the database, in the tracked person/people’s records.
  #### 3. Software system for each node: To be installed in individual systems or ‘nodes’. Here, authorized personnel will perform operations. As per login credentials, access rights will be defined for the user.
CCTV surveillance system provides the real-time recording as well as online access by the staff to  monitor  more intelligently that would eventually reduce the risk of crime.
 CCTV surveillance system provides the real-time recording as well as online access by the staff to  monitor  more intelligently that would eventually reduce the risk of crime.
  ## WORKFLOW 
  ![WORK FLOW](/demo/Screenshot.png)
  
### Prerequisites

The major part of this project is done on Tensorflow. The aquisition of video is done using OpenCV and the GUI is produced using PyQt4.
To run this project, the system is only required to have a [Python 3.6.x](https://www.python.org/downloads/release/python-365/) interpreter installed.

### Installing

For realtime performance, Tensorflow GPU is also required. Further details on [Tensorflow.org](https://www.tensorflow.org/install/gpu)
Detailed system requirements are produced in ```requirements.txt``` 
After cloning the repository, at root directory :
```
pip install -r requirements.txt
```

## Usage

For Linux Systems :
```python3 gui.py``` (for GUI version)
```python3 yolo.py``` (for non-GUI version)

For Windows Systems :
```python3 gui.py``` (for GUI version)
```python3 yolo.py``` (for non-GUI version)

## Built With

* [Tensorflow]() - The DL framework used
* [Python]() - Language used
* [PyQt]() - Used for producing GUI
* [Opencv]() - Used for video aquisition

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details
