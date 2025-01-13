# Machine Learning Project: Bin Packing Problem

**Authors: 黄靖远, 沈嘉轩, 周嘉宏**

## Core Codes

- ``PackingUtils.py``: packing utils, including placement validation check, height map update, and most importantly, the EMS algorithm and placement generation implementation.
- ``ModelUtils.py``: Some model blocks. The ``CNN_Encoder`` is used for encoding the height map.
- ``ModelEMS.py``: Core functions of the Policy Network, including transformer encoders, cross-attention blocks, and several fully-connected layers. ``BPP_Model_EMS`` is the main model class.
- ``train_reinforce.py``: Core training code for the Policy Network.
- ``data_generation.ipynb``: Generates the training data for the Policy Network, i.e. the items to be packed.
- ``test_model_visualize.ipynb``: Tests and Visualizes the packing results of the trained Policy Network.
- ``dfs_search_jiahong.ipynb``: Implementations of the DFS algorithm for single-container bin packing problem.
- ``dfs_multibin.ipynb``: Successful implementation of the DFS algorithm for the bin packing problem of multiple containers.
- ``models/``: Folder of trained models. 