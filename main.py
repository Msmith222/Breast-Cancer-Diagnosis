# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(MichaelSmith)s
"""

#import urllib to download csv, pandas for dataframe and numpy for data manipulation and matplotlib for histograms,pdfpages saves all histograms in one pdf file..
from urllib.request import urlopen
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def phase1():
    #url request brings in data.
    response = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    
    #a dataframe is created from data pulled in.
    df = pd.read_csv(response, header = None, names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11"])
    
    #missing data converted to NaN, then converted to a float column, then the average for that column is inserted into NaN.
    df.replace('?', np.nan, inplace=True) 
    df['A7'] = df['A7'].astype(float)
    df["A7"].fillna(df["A7"].mean(), inplace=True)  
    
    #a new dataframe is created to put the mean, median, standard deviation, and variance of each column into a new dataframe
    stats = pd.DataFrame(columns = ['Mean', 'Median', 'Standard Deviation', 'Variance'])
    mean = []
    median = []
    std = []
    var = []
    
    #this loop will calculate various statistical values of each column in the dataframe.
    for i in range(1):
        mean.append(df.mean())
        median.append(df.median())
        std.append(df.std())
        var.append(df.var())
    
    #the corresponding data inserted into each column.
    stats['Mean'] = mean[0]
    stats['Median'] = median[0]
    stats['Standard Deviation'] = std[0]
    stats['Variance'] = var[0]
    
    #A1 and A11 are dropped from the dataframe.
    stats = stats.drop('A1')
    stats = stats.drop('A11')
    
    #dataframe is printed.
    print(stats)

    #mean, median, standard deviation and variance for A2-A11 written to stats.csv.
    stats.to_csv('stats.csv', sep = ',')
    
    
    #9 subplots of a figure will be utilized to create 9 graphs for A2-A10.
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6 = plt.figure()
    fig7 = plt.figure()
    fig8 = plt.figure()
    fig9 = plt.figure()
    
    sp1 = fig1.add_subplot(1,1,1)
    sp2 = fig2.add_subplot(1,1,1)
    sp3 = fig3.add_subplot(1,1,1)
    sp4 = fig4.add_subplot(1,1,1)
    sp5 = fig5.add_subplot(1,1,1)
    sp6 = fig6.add_subplot(1,1,1)
    sp7 = fig7.add_subplot(1,1,1)
    sp8 = fig8.add_subplot(1,1,1)
    sp9 = fig9.add_subplot(1,1,1)
    
    
    #histograms of varying  colors for A2-A10 plotted into the subplots of figure 1, axes and title are specified.
    sp1.hist(df['A2'], bins = 10, color = 'yellow')
    sp1.set_title('Clump Thickness') 
    sp1.set_ylabel('Frequency') 
    sp1.set_xlabel('Thickness')
    
    sp2.hist(df['A3'],bins = 10, color = 'blue')
    sp2.set_title('Uniformity of Cell Size') 
    sp2.set_ylabel('Frequency') 
    sp2.set_xlabel('Size')
    
    sp3.hist(df['A4'],bins = 10, color = 'red')
    sp3.set_title('Uniformity of Cell Shape') 
    sp3.set_ylabel('Frequency') 
    sp3.set_xlabel('Uniformity of Shape')
    
    sp4.hist(df['A5'],bins = 10, color = 'orange')
    sp4.set_title('Marginal Adhesion') 
    sp4.set_ylabel('Frequency') 
    sp4.set_xlabel('Adhesion')
    
    sp5.hist(df['A6'],bins = 10, color = 'pink')
    sp5.set_title('Single Epithelial Cell Size') 
    sp5.set_ylabel('Frequency') 
    sp5.set_xlabel('Cell Size')
    
    sp6.hist(df['A7'],bins = 10, color = 'green')
    sp6.set_title('Bare Nuclei') 
    sp6.set_ylabel('Frequency') 
    sp6.set_xlabel('Bare Nuclei')
    
    sp7.hist(df['A8'],bins = 10, color = 'purple')
    sp7.set_title('Bland Chromatin') 
    sp7.set_ylabel('Frequency') 
    sp7.set_xlabel('Bland Chromatin')
    
    sp8.hist(df['A9'],bins = 10, color = 'grey')
    sp8.set_title('Normal Nucleoli') 
    sp8.set_ylabel('Frequency') 
    sp8.set_xlabel('Normal Nucleoli')
    
    sp9.hist(df['A10'],bins = 10, color = 'black')
    sp9.set_title('Mitoses') 
    sp9.set_ylabel('Frequency') 
    sp9.set_xlabel('Mitoses')
    
    pp = PdfPages('histograms.pdf')
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.savefig(fig5)
    pp.savefig(fig6)
    pp.savefig(fig7)
    pp.savefig(fig8)
    pp.savefig(fig9)
    pp.close()

def phase2():
    response = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    
    #a dataframe is created from data pulled in.
    df = pd.read_csv(response, header = None, names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11"])
    
    #missing data converted to NaN, then converted to a float column, then the average for that column is inserted into NaN.
    df.replace('?', np.nan, inplace=True) 
    df['A7'] = df['A7'].astype(float)
    df["A7"].fillna(df["A7"].mean(), inplace=True)  
    
    #convert df to numpy array
    array = df.as_matrix()
    
    #remove lable and ID from array.
    array1 = array[:,1:10]
    #randomly choose two values
    a1,a2 = np.random.randint(698,size= 2)
    
    #initialize array for each values and its grouped label.
    recalc2 = []
    recalc4 = []
    
    #randomly selected numbers choose a row of the array as initial centroids.
    u2 = array1[a1 ,:]
    u4 = array1[a2,:]
    
    #repeat u2,u4 determination 1500 iterations.   
    for y in range (1500):
        #loop through array and determine each row at label 2 or 4.
        for i in range(0,699):
            x = array1[i,:]  
            z1 = np.sqrt(sum((x-u4)**2))
            z2 = np.sqrt(sum((x-u2)**2))
            
            #pass each row to either recalc4 or recalc2 depending on calculated euclidean distance.
            if z1 < z2:
                recalc4.append(array1[i,:])
            else:
                recalc2.append(array1[i,:])
                
        #recalculate the mean u2, u4 based on values grouped into recalc2 or recalc4.
        u2 = np.mean(recalc2, axis = 0)
        u4 = np.mean(recalc4, axis = 0)
        
    print("-----------------------Final Mean----------------------------")
    
    #final calculated mean based on results.
    print("mu_2:",u2)
    print("mu_4:",u4)
    
    #pass all data rows through final u2,u4 calulation to determine class label as 4 or 2.
    prediction = []
    for i in range(0,699):
        x = array1[i,:]  
        z1 = np.sqrt(sum((x-u4)**2))
        z2 = np.sqrt(sum((x-u2)**2))
            
        if z1 < z2:
            prediction.append(4)
            
        elif z1 > z2:
            prediction.append(2)
    
    #create a new data frame for ID, Class and the predicted result from the above loop.
    cluster = pd.DataFrame(columns = ['ID', 'Class' , 'Predicted Class'])
    
    cluster['ID'] = df['A1']
    cluster['Class'] = df['A11']
    cluster['Predicted Class'] = prediction
    
    print("---------------------Cluster Assignment-----------------------")
    print(cluster)
    
    #dataframe cluster saved as a csv.
    cluster.to_csv('Cluster_Assignment.csv', sep = ',')
    
def phase3():
     #import cluster_assignment results from phase2() as a df.
    df = pd.read_csv("Cluster_Assignment.csv")
    array = df.as_matrix()
    
    #initialization of variables for loop.
    errorB = 0
    errorM = 0
    predicted4 = 0
    predicted2 = 0
    
    #each row is looped to determine which predictions were incorrect and count total predictions of each outcome.
    for i in range(0,699):
        if array[i,2] == 2 and array[i,3] == 4:
            errorB = errorB + 1
        if array[i,3] == 4:
            predicted4 = predicted4 + 1
        if array[i,2] == 4 and array[i,3] == 2:
            errorM = errorM + 1
        if array[i,3] ==2:
            predicted2 = predicted2 + 1
    
    
    #calculate error based on results from previous for loop.
    Benign_error = errorB/predicted2
    Malignant_error = errorM/predicted4
    
    #user is given back error B, error M and total error rate.
    print("Error B =", Benign_error)
    print("Error M =", Malignant_error)
    print("Total error rate =", Benign_error + Malignant_error )
    
def main():
    
    #each phase of the project is assigned within its own function and called in order.
    phase1()
    phase2()
    phase3()
    
main()

    