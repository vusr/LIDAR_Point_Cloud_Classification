# Seoul Robotics Assignment

## Task Description

Thank you for your interest in joining us! This assignment provides you with clustered point cloud data classified into four distinct classes: **car**, **pedestrian**, **bicyclist**, and **background**. Your primary task is to develop a robust **classification model** and implement an effective **evaluation process** for assessing its performance.

The train and test datasets are structured into subdirectories for each class (**background**, **bicyclist**, **car**, and **pedestrian**) and are located in the following directories:
- **Train**: `data/train`
- **Test**: `data/test`.

To load point cloud data from a binary file containing x, y, z coordinates in Python, you can use the `numpy` package as shown below:

```python
points = numpy.fromfile(bin_file, dtype=np.float32).reshape(-1, 3)
```


## Optional Challenge

If the task is too easy for you, we invite you to try a more challenging task. The data for this challenge is located in the `data/optional_challenge_data` folder. This dataset contains point clouds for **10 sequential scenes** without pre-segmentation.

### Your Task

1. Segment each frame into five classes: **car**, **pedestrian**, **bicyclist**, **ground**, and **background**.
2. Perform **clustering** and **tracking** for the **car**, **pedestrian**, and **bicyclist** classes.

In this dataset, we also provide **intensity** and **ring** values. You can read the data using the following `numpy` snippet:

```python
points = numpy.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
```


## Requirements

1. **Programming Language**: Use **Python**, **C++**, or both for this assignment.
2. **Code Quality**: Submit a clear and well-organized code in a properly structured repository. We also expect you to provide instructions for running your code.
3. **Report**: Alongside your code, please provide a report explaining your methodology, evaluation process, and a discussion. The report can be in one of the following formats: **PDF** or **Markdown**.

_Note:_  Please **DO NOT** share the data, we want to have a fair recruitment process.


## Data Visualization

We provide a script `data_visualize.py` to help you visualize the point cloud data. The script uses **VisPy** to render the 3D point clouds interactively.

### Usage

1. Install the required packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Set the `data_folder` variable in `data_visualize.py` to the folder containing the point cloud files.
3. Run the script:
   ```bash
   python data_visualize.py
   ```
4. Navigate through frames by using the interactive control:
  - Press `N` to move to the next frame.
  - Press `B` to move to the previous frame.

---

**Have fun!**

