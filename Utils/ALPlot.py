#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#import matplotlib
#matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import datetime
import numpy as np
import os
import re
import sys

class Ploter(object):

    def __init__(self,data=None, path=None):
        self.data = data
    
        if os.path.isdir(path):
            self.path = path
        else:
            self.path = None

    def _draw_train_data(self,title=''):

        fig = plt.figure(1)
        fig.suptitle(title)
        fig.subplots_adjust(top=0.3)
        
        #Train size x Acquisition step time (if that was logged)
        if len(self.data['time']) > 0:
            plt.subplot(211)
            plt.plot(self.data['trainset'],self.data['time'],'bo')
            plt.axis([self.data['trainset'][0]-100,self.data['trainset'][-1:][0]+100,0.1,self.data['time'][-1:][0]+0.5])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.gcf().autofmt_xdate()
            plt.xlabel('Train set size')
            plt.ylabel('Acquisition step time (hours)')

        #Train size x AUC
        if len(self.data['auc']) > len(self.data['trainset']):
            self.data['auc'].pop()
            
        plt.subplot(212)
        min_auc = np.asarray(self.data['auc']).min()
        plt.plot(self.data['trainset'],self.data['auc'],'k-')
        plt.axis([self.data['trainset'][0],self.data['trainset'][-1:][0],min_auc-0.1,1.0])
        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Train set size')
        plt.ylabel('AUC')

        plt.tight_layout()
        plt.show()


    def plotFromExec(self,data):
        pass
    
    def plotSlurm(self,path=None):

        if path is None:
            path = self.path

        dir_contents = os.listdir(path)
        slurm_path = None
        for fi in dir_contents:
            if fi.startswith('slurm'):
                slurm_path = os.path.join(path,fi)
            
        self.data = {'time':[],
                'auc':[],
                'trainset':[]}
        start_line = 0
        timerex = r'Acquisition step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        aucrex = r'AUC: (?P<auc>0.[0-9]+)'
        trainsetrex = r'Train set: (?P<set>[0-9]+) items'
        
        timerc = re.compile(timerex)
        aucrc = re.compile(aucrex)
        trainrc = re.compile(trainsetrex)

        with open(slurm_path,'r') as fd:
            lines = fd.readlines()

        #Set a time reference
        for line in lines:
            lstrip = line.strip()
            tmatch = timerc.fullmatch(lstrip)
            aucmatch = aucrc.fullmatch(lstrip)
            trmatch = trainrc.fullmatch(lstrip)
            if tmatch:
                td = datetime.timedelta(hours=int(tmatch.group('hours')),minutes=int(tmatch.group('min')),seconds=round(float(tmatch.group('sec'))))
                self.data['time'].append(td.total_seconds()/3600.0)
            if aucmatch:
                self.data['auc'].append(float(aucmatch.group('auc')))
            if trmatch:
                self.data['trainset'].append(int(trmatch.group('set')))

        self._draw_train_data('SLURM log')

if __name__ == "__main__":
    dataset = str(input("Enter dataset path: "))

    p = Ploter(path=dataset)
    p.plotSlurm()
    
