# readme

​      Our code runs under **python 3.7** and **tensorflow 1.14.0**.

​      We proposed a method for drug-drug interaction prediction called *MDAE*. The file contains the code to implement the method and data on the interactions between 2367 drugs.

​      When using this method, you need to load all the files in the folder into your running environment. There are three program files in the folder. When you use it, you only need to adjust some parameters in the following line of code in **KCV_new.py**:

​        *CV_num, method_num, alpha, beta, gama, mu, dimension, drug_size, removed_ratio, learning_rate=[3, 4,0.1,2,0.1,1e-5,128,2367,0,0.001]*

​        Actually this is an assignment statement. *CV_num* refers to the fold-number of cross-validation,*Method_num* refers to the method involved in cross-validation. *Alpha, beta, gama, mu* and *dimension* are parameters in the model we proposed. *Drug_size* is the number of drugs in your data. *Removed_ratio* and *learning_rate* just mean the ratio of remove link and the learning rate.