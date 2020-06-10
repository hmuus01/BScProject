**This computing project aims at building a fraud detection service able to robustly and automatically identify fraudulent online credit card transactions.**

| **Main Files** |**Description** |**COMMENTED/NOT COMMENTED** |
| --- | --- | --- |
|`LR_Performance.py`| This file contains the steps taken to **train the model and test the performance** of the LR model | ✔ |
|`RF_Performance.py`| This file contains the steps taken to **train the model and test the performance** of the RF model | ✔ |
|`SVM_Performance.py`|This file contains the steps taken to **train the model and test the performance** of the SVM model | ✔ |
|`LR_Save_Model.py`| This file contains the steps taken to save the LR model for future use after optimal parameters were found from testing.|  ✔ |
|`RF_Save_Model.py`| This file contains the steps taken to save the RF model for future use after optimal parameters were found from testing.|  ✔ |
|`SVM_Save_Model.py`| This file contains the steps taken to save the SVM model for future use after optimal parameters were found from testing.|  ✔ |
|`server.py`|This File contains the code for the server which uses the three models in this project to predict the class of a transaction | ✔ |
|`flaskapp.py` | This file serves as the client, it is made with flask and receives data from the server which is displayed to the user. | ✔ |

| **Folders In This Project** |**Description** |**COMMENTED/NOT COMMENTED** |
| --- | --- | --- |
|Data| This Folder contains the datasets used in this project. **Please note the only two files used in this project are `dev_data` used in training and testing and `validation_data` used for the client/server web application**| ✔ |
|LBR Tests |This folder contains the steps taken to train the model and test the performance of the model using `20 different lBR values` and analysing the performance| ✔ |
|SelectKBest Tests| This folder contains the steps taken to train the model and test the ML models with different `SelectKBest functions` on all the features and how this impacts the performance is analysed | ✔ |
|Threshold Tests | This folder contains the steps taken to train the model and test the performance of the 4 metrics precision, recall, f1-score and accuracy on different threshold values - range(0.0-0.95) | ✔ |
|Probability Distribution | This folder contains the files to plot the  `probability prediction distribtuion` graphs for the three algorithms RF, LR, SVM | ✔ |
|Recall Distribution | This folder contains the files to plot the `recall distribtuion` graphs for the three algorithms RF, LR, SVM | ✔ |
|Accuracy Distribution | This folder contains the files to plot the `accuracy distribtuion` graphs for the three algorithms RF, LR, SVM | ✔ |
|Models | This folder contains the `final models` from training and testing e.g the models with final LBR and best Features | ✔ |
|Templates | These files contain the html code displayed to the user on the client side | ✔ |
|Static | The folder contains the file with the css(styling) code displayed to the user on the client side | ✔ |
|Unused Files | This folder contains unused files not used in this project and are just failed attempted test files. | ✔ |