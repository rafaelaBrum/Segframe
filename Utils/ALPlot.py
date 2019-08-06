#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#import matplotlib
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import datetime
import numpy as np
import os
import re
import sys

class Ploter(object):

    def __init__(self,data=None, path=None):
        if not path is None and os.path.isdir(path):
            self.path = path
        else:
            self.path = None

    def draw_data(self,data,title=''):

        fig = plt.figure(1)
        fig.suptitle(title)
        fig.subplots_adjust(top=0.3)
        
        #Train size x Acquisition step time (if that was logged)
        if data['time'].shape != data['trainset'].shape:
            maxi = min(data['time'].shape[0],data['trainset'].shape[0])
        else:
            maxi = max(data['time'].shape[0],data['trainset'].shape[0])

        if data['time'].shape[0] > 0:
            plt.subplot(211)
            plt.plot(data['trainset'][:maxi],data['time'][:maxi],'bo')
            plt.axis([data['trainset'][0]-100,data['trainset'].max()+100,0.0,data['time'].max()+.2])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.gcf().autofmt_xdate()
            plt.xlabel('Train set size')
            plt.ylabel('Acquisition step time (hours)')

        #Train size x AUC
        if data['auc'].shape[0] > data['trainset'].shape[0]:
            print("AUC results are too many")
            print(data['auc'])
            print(data['trainset'])
            data['auc'] = data['auc'][:-1]

        if data['auc'].shape[0] > 0:
            plt.subplot(212)
            min_auc = data['auc'].min()
            plt.plot(data['trainset'],data['auc'],'k-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Train set size')
            plt.ylabel('AUC')
        elif data['accuracy'].shape[0] > 0:
            plt.subplot(212)
            min_auc = data['accuracy'].min()
            plt.plot(data['trainset'],data['accuracy'],'k-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('Train set size')
            plt.ylabel('Accuracy')
            
        plt.tight_layout()
        plt.show()

    def draw_multiline(self,data,title):

        palette = plt.get_cmap('Set1')

        color = 0
        plotAUC = False
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        for k in data:
            if data[k]['auc'].shape[0] > 0:            
                plt.plot(data[k]['trainset'],data[k]['auc'], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                color += 1
                plotAUC = True
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(data[k]['auc'].min())
                max_y.append(data[k]['auc'].max())           
            else:
                plt.plot(data[k]['trainset'],data[k]['accuracy'], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                color += 1
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())                
                min_y.append(data[k]['accuracy'].min())
                max_y.append(data[k]['accuracy'].max())
                
        plt.legend(loc=4,ncol=2)
        plt.xticks(np.arange(100, max(max_x)+1, 100.0))
        plt.yticks(np.arange(min(min_y), 1.0, 0.06))
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Training set size")
        if plotAUC:
            plt.ylabel("AUC")
        else:
            plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def plotFromExec(self,data):
        pass

    def parseResults(self,path,al_dirs):
        data = {}

        for d in al_dirs:
            d_path = "{0}-{1}".format(path,d)
            if os.path.isdir(d_path):
                data[d] = self.parseSlurm(d_path)

        return data
    
    def parseSlurm(self,path=None):

        if path is None and self.path is None:
            print("No directory found")
            sys.exit(-1)
        elif path is None:
            path = self.path

        dir_contents = os.listdir(path)
        slurm_path = None
        for fi in dir_contents:
            if fi.startswith('slurm'):
                slurm_path = os.path.join(path,fi)

        if slurm_path is None:
            print("No slurm file in path: {0}".format(path))
            return None
        
        data = {'time':[],
                'auc':[],
                'trainset':[],
                'accuracy':[]}
        start_line = 0
        timerex = r'Acquisition step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        aucrex = r'AUC: (?P<auc>0.[0-9]+)'
        accrex = r'Accuracy: (?P<acc>0.[0-9]+)'
        trainsetrex = r'Train set: (?P<set>[0-9]+) items'
        
        timerc = re.compile(timerex)
        aucrc = re.compile(aucrex)
        trainrc = re.compile(trainsetrex)
        accrc = re.compile(accrex)

        with open(slurm_path,'r') as fd:
            lines = fd.readlines()

        #Set a time reference
        for line in lines:
            lstrip = line.strip()
            tmatch = timerc.fullmatch(lstrip)
            aucmatch = aucrc.fullmatch(lstrip)
            trmatch = trainrc.fullmatch(lstrip)
            accmatch = accrc.fullmatch(lstrip)
            if tmatch:
                td = datetime.timedelta(hours=int(tmatch.group('hours')),minutes=int(tmatch.group('min')),seconds=round(float(tmatch.group('sec'))))
                data['time'].append(td.total_seconds()/3600.0)
            if aucmatch:
                data['auc'].append(float(aucmatch.group('auc')))
            if trmatch:
                data['trainset'].append(int(trmatch.group('set')))
            if accmatch:
                data['accuracy'].append(float(accmatch.group('acc')))

        #Use NP arrays
        data['time'] = np.asarray(data['time'])
        data['auc'] = np.asarray(data['auc'])
        data['trainset'] = np.asarray(data['trainset'])
        data['accuracy'] = np.asarray(data['accuracy'])

        if data['auc'].shape[0] > 0:
            print("Min AUC: {0}; Max AUC: {1}".format(data['auc'].min(),data['auc'].max()))            
        if data['accuracy'].shape[0] > 0:
            print("Min accuracy: {0}; Max accuracy: {1}".format(data['accuracy'].min(),data['accuracy'].max()))

        return data

if __name__ == "__main__":

    if len(sys.argv) > 1:
        p = Ploter()
        if os.path.isdir(os.path.dirname(sys.argv[1])):
            if len(sys.argv) >= 3:
                plot_dirs = sys.argv[2].split(',')
            else:
                plot_dirs = str(input("Enter AL dir numbers to plot (comma separated): ")).split(',')
            data = p.parseResults(sys.argv[1],plot_dirs)
            if len(sys.argv) == 4:
                title = sys.argv[3]
            else:
                title = 'AL Experiment'

            p.draw_multiline(data,title)
        else:
            print("First argument should be a directory path where AL results are.")
            sys.exit(-1)
            
    else:
        dataset = str(input("Enter dataset path: "))

        p = Ploter(path=dataset)
        p.draw_data(p.parseSlurm(),'SLURM log')

    
