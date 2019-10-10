import numpy as np
import csv
import random
import sys
import math
sys.path.append("..")
import graph, mdae
import copy
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from gensim.models import Word2Vec, KeyedVectors
import datetime
from sklearn.ensemble import RandomForestClassifier
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def calculate_metric_score(real_labels,predict_score):
   precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
   aupr_score = auc(recall, precision)
   all_F_measure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + recall[k]) > 0:
          all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          all_F_measure[k] = 0
   print("all_F_measure: ")
   print(all_F_measure)
   max_index = all_F_measure.argmax()
   threshold = pr_thresholds[max_index]
   fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
   auc_score = auc(fpr, tpr)


   f = f1_score(real_labels, predict_score)
   print("f-score:"+str(f))
   accuracy = accuracy_score(real_labels, predict_score)
   precision = precision_score(real_labels, predict_score)
   recall = recall_score(real_labels, predict_score)
   print('results for feature:' + 'weighted_scoring')
   print(    '************************AUPR score:%.3f,  AUC score:%.3f, f score:%.3f,accuracy:%.3f, precision score:%.3f, recall score:%.3f************************' % (
        aupr_score,  auc_score, f, accuracy, precision,  recall))
   results = [aupr_score,  auc_score, f, accuracy, precision,  recall]

   return results


def cross_validation(drug_drug_matrix, CV_num, method_num, alpha, beta, gama, mu, dimension, drug_size, learning_rate):
    results = []
    link_number = 0
    nonLinksPosition = []  # all non-link position
    link_position = []
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):

            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])
    all_position = np.array(link_position)
    print("all_position:" + str(len(all_position)))
    nonLinksPosition = np.array(nonLinksPosition)

    index = np.arange(0, len(all_position))
    random.shuffle(index)

    g = graph.Graph()
    g.read_edgelist("all.txt")
    print(g.G.number_of_edges())

    fold_num = len(all_position) // CV_num
    print(fold_num)
    for CV in range(0, CV_num):
        print('*********round:' + str(CV) + "**********\n")
        starttime = datetime.datetime.now()
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        train_index  = np.setdiff1d(index, test_index)
        test_index.sort()
        train_index.sort()
        testPosition = np.array(all_position)[test_index]
        trainPosition = np.array(all_position)[train_index]

        for i in range(0, len(testPosition)):
            if drug_drug_matrix[testPosition[i, 0]][testPosition[i, 1]] == 1:
                g.G.remove_edge(str(testPosition[i, 0] + 1), str(testPosition[i, 1] + 1))
        print(g.G.number_of_edges())
		
        print("Test Begin")
        model = mdae.MDAE(g,[1000, dimension],alpha, beta, gama, mu, drug_size, learning_rate)
        print("Test End")

        X_train = []
        Y_train = []
        trainPosition = np.concatenate((np.array(trainPosition), nonLinksPosition), axis=0)
        for i in range(0, len(trainPosition)):
            if method_num==1 :
            # 1: Average
                X_train.append((model.vectors[str(trainPosition[i, 0] + 1)] + model.vectors[str(trainPosition[i, 1] + 1)]) / 2)
            elif method_num==2:
            # 2: Hadamard
                X_train.append(model.vectors[str(trainPosition[i, 0] + 1)] * model.vectors[str(trainPosition[i, 1] + 1)])
            elif method_num==3:
            # 3: Weighted-L1
                X_train.append(np.fabs(model.vectors[str(trainPosition[i, 0] + 1)] - model.vectors[str(trainPosition[i, 1] + 1)]))
            elif method_num==4:
            # 4: Concatenate
                X_train.append(np.concatenate((model.vectors[str(trainPosition[i,0]+1)],model.vectors[str(trainPosition[i,1]+1)]),axis=0))
            Y_train.append(drug_drug_matrix[trainPosition[i,0], trainPosition[i,1]])

        classifier = RandomForestClassifier(n_estimators=10)
        classifier.fit(X_train, Y_train)
        X_test= []
        Y_test = []
        testPosition = np.concatenate((np.array(testPosition), nonLinksPosition), axis=0)
        for i in range(0, len(testPosition)):
            if method_num==1 :
            # 1: Average
                X_test.append((model.vectors[str(testPosition[i, 0] + 1)] + model.vectors[str(testPosition[i, 1] + 1)]) / 2)
            elif method_num==2:
            # 2: Hadamard
                X_test.append(model.vectors[str(testPosition[i, 0] + 1)] * model.vectors[str(testPosition[i, 1] + 1)])
            elif method_num==3:
            # 3: Weighted-L1
                X_test.append(np.fabs(model.vectors[str(testPosition[i, 0] + 1)] -model.vectors[str(testPosition[i, 1] + 1)]))
            elif method_num==4:
            # 4: Concatenate
                X_test.append(np.concatenate((model.vectors[str(testPosition[i,0]+1)],model.vectors[str(testPosition[i,1]+1)]),axis=0))
            Y_test.append(drug_drug_matrix[testPosition[i, 0], testPosition[i, 1]])
        y_pred_label = classifier.predict(X_test)

        results.append(calculate_metric_score(Y_test, y_pred_label))
        endtime = datetime.datetime.now()
        print(endtime - starttime)
    return results

#########################################################################################################
#########################################################################################################
    
CV_num, method_num, alpha, beta, gama, mu, dimension, drug_size, removed_ratio, learning_rate=[3, 4,0.1,2,0.1,1e-5,128,2367,0,0.001]

#########################################################################################################
#########################################################################################################
# number of Drugs(DDI>5) is 2367
adj = np.zeros((drug_size, drug_size),dtype=np.int8)
j=0
k=0
np.set_printoptions(suppress=True)
with open("./data/all.csv", "rt", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        if int(i[0])<=drug_size and int(i[1])<=drug_size:
            adj[int(i[0])-1,int(i[1])-1] = 1
            j = j + 1
        k = k + 1
    print(np.sum(adj==1))
print(adj)
print(j)
print(k)
adj = mdae.miss_edges(adj,removed_ratio)
results = cross_validation(adj, CV_num, method_num, alpha, beta, gama, mu, dimension, drug_size, learning_rate)
print(results)
results = np.array(results)
print(results.sum(axis=0)/CV_num)


