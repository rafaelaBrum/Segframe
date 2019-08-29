#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#import matplotlib
#matplotlib.use('macosx')
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import datetime
import numpy as np
import os
import re
import sys
import argparse

class Plotter(object):

    def __init__(self,data=None, path=None):
        if not path is None and os.path.isdir(path):
            self.path = path
        else:
            self.path = None

    def draw_uncertainty(self,data,xticks,spread=1,title=''):
        """
        Data: list of tuples in the form (indexes,uncertainties), where both indexes and uncertainties are numpy
        arrays.
        """
        mpl.rcParams['agg.path.chunksize'] = 1000000
        n_points = 20000
        plots = []
        maxu = 0.0
        pl1,pl2 = None,None
        for k in range(len(data)):
            indexes,unc = data[k]
            selected = unc[indexes]
            unselected = np.delete(unc,indexes)
            mk = np.max(unselected)
            if mk > maxu:
                maxu = mk
            pl1 = plt.plot(np.random.rand(selected.shape[0])*spread+xticks[k],selected, 'oc',alpha=0.6)
            pl2 = plt.plot(np.random.rand(n_points)*spread+xticks[k],np.random.choice(unselected,n_points),
                         'oy',markersize=2,alpha=0.5)
            
        plots.append(pl1)
        plots.append(pl2)

        labels = ['Selected images','Unselected images']
        plt.legend(plots,labels=labels,loc=4,ncol=2)
        plt.xticks(xticks)
        plt.yticks(np.arange(0.0, maxu+0.1, 0.2))
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Acquisition #")
        plt.ylabel("Uncertainty")

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def draw_stats(self,data,xticks,auc_only,labels=None,confidence=2,spread=1,title=''):
        """
        @param data <list>: a list as returned by calculate_stats
        """
        palette = plt.get_cmap('Set1')
        color = 0
        plots = []
        for d in data:
            x_data,y_data,dev,y_label = d
            if auc_only:
                ax = plt
            else:
                _,ax = plt.subplots()
            plots.append(ax)
            ax.plot(x_data, y_data, lw = 1, color = palette(color), alpha = 1)
            # Shade the confidence interval
            low_ci = y_data - confidence*dev
            upper_ci = y_data + confidence*dev
            ax.fill_between(x_data, low_ci, upper_ci, color = palette(color), alpha = 0.4)
            color += 1
            if auc_only:
                ax.xlabel("Trainset size")
                ax.ylabel(y_label)
            else:
                ax.set_xlabel("Trainset size")
                ax.set_ylabel(y_label)
                
        # Label the axes and provide a title
        if labels is None:
            labels = ['Mean','{} STD'.format(confidence)]
        plt.legend(plots,labels=labels,loc=4,ncol=2)
        plt.xticks(np.arange(min(x_data), max(x_data)+xticks, xticks))
        plt.yticks(np.arange(np.min(low_ci)-0.05, np.max(upper_ci)+0.05, 0.1))
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')            
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
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

    def draw_multiline(self,data,title,xtick):

        palette = plt.get_cmap('Set1')

        color = 0
        plotAUC = False
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        for k in data:
            if 'auc' in data[k] and data[k]['auc'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['auc'].shape[0]:
                    data[k]['auc'].append(data[k]['auc'][-1])
                    
                plt.plot(data[k]['trainset'],data[k]['auc'], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                color += 1
                plotAUC = True
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(data[k]['auc'].min())
                max_y.append(data[k]['auc'].max())
                print(data[k]['trainset'])
            else:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['accuracy'].shape[0]:
                    data[k]['accuracy'].append(data[k]['accuracy'][-1])
                    
                plt.plot(data[k]['trainset'],data[k]['accuracy'], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                color += 1
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())                
                min_y.append(data[k]['accuracy'].min())
                max_y.append(data[k]['accuracy'].max())
                
        plt.legend(loc=4,ncol=2)
        plt.xticks(np.arange(min(min_x), max(max_x)+1, xtick))
        if max(max_x) > 1000:
            plt.xticks(rotation=30)
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

    def extractNPData(self,path):
        if not os.path.isfile(path):
            print("Given file path is incorrect. Check that.")
            sys.exit(1)

        data = np.load(path)
        return data

    def generateIndexes(self,size,start,step):
        """
        Should be used when X values are not given but are a sequence of fixed period integers
        """
        return np.asarray(range(start,(size*step)+start,step))
                              
    def parseSlurm(self,path=None):

        if path is None and self.path is None:
            print("No directory found")
            sys.exit(1)
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


    def retrieveUncertainty(self,config):
        unc_files = []
        import pickle

        if not config.all:
            for i in config.ac_n:
                unc_file = 'al-uncertainty-{}-r{}.pik'.format(config.ac_func,i)
                if os.path.isfile(os.path.join(config.sdir,unc_file)):
                    unc_files.append(unc_file)
        else:
            items = os.listdir(config.sdir)            
            for f in items:
                if f.startswith('al-uncertainty'):
                    unc_files.append(f)

        data = []

        for f in unc_files:
            with open(os.path.join(config.sdir,f),'rb') as fd:
                indexes,uncertainties = pickle.load(fd)
            data.append((indexes,uncertainties))

        return data

    def calculate_stats(self,data,auc_only):
        """
        @param data <dict>: a dictionary as returned by parseResults

        Calculates mean and standard deviation for AUC and/or Accuracy.

        Returns a list of tuples (trainset,mean_values,std dev) for each AUC and Accuracy
        """
        auc_value = None
        acc_value = None
        i = 0
        trainset = None
        
        for k in data:
            if 'auc' in data[k] and data[k]['auc'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['auc'].shape[0]:
                    data[k]['auc'].append(data[k]['auc'][-1])
                if auc_value is None:
                    trainset = data[k]['trainset']
                    shape = (len(data),len(trainset))
                    auc_value = np.ndarray(shape=shape,dtype=np.float32)
                auc_value[i] = data[k]['auc']
            if not auc_only and 'accuracy' in data[k] and data[k]['accuracy'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['accuracy'].shape[0]:
                    data[k]['accuracy'].append(data[k]['accuracy'][-1])
                if auc_value is None:
                    shape = (len(data),len(data[k]['trainset']))
                    acc_value = np.ndarray(shape=shape,dtype=np.float32)
                acc_value[i] = data[k]['accuracy']

            i += 1
        #Return mean and STD dev
        if auc_only:
            return [(trainset,np.mean(auc_value.transpose(),axis=1),np.std(auc_value.transpose(),axis=1),"AUC")]
        else:
            return [(trainset,np.mean(arr[0].transpose(),axis=1),np.std(arr[0].transpose(),axis=1),arr[1]) for arr in ((auc_value,"AUC"),
                                                                                                              (acc_value,"Accuracy"))]
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    ##Multiline SLURM parse
    parser.add_argument('--multi', action='store_true', dest='multi', default=False, 
        help='Plot multiple lines from slurm files.')
    parser.add_argument('-r', dest='results', type=str,default='results', 
        help='Directory containing results.')
    parser.add_argument('-ids', dest='ids', nargs='+', type=int, 
        help='Experiment IDs to plot.', default=None,required=False)
    parser.add_argument('-xtick', dest='xtick', nargs=1, type=int, 
        help='xtick interval.', default=200,required=False)    
    parser.add_argument('-t', dest='title', type=str,default='AL Experiment', 
        help='Figure title.')
    parser.add_argument('-type', dest='tmode', type=str, 
        help='Experiment type: \n \
        AL - General active learning experiment; \n \
        MN - MNIST dataset experiment.',
       choices=['AL','MN'],default='AL')
    
    ##Single experiment plot
    parser.add_argument('--single', action='store_true', dest='single', default=False, 
        help='Plot data from a single experiment.')
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path (should contain an slurm file).')

    ##Make stats
    parser.add_argument('--stats', action='store_true', dest='stats', default=False, 
        help='Make mean and STD dev from multiple runs.')
    parser.add_argument('-auc', action='store_true', dest='auc_only', default=True, 
        help='Calculate statistics for AUC only.')
    parser.add_argument('-ci', dest='confidence', nargs=1, type=int, 
        help='CI.', default=2,required=False)
    parser.add_argument('-n', dest='n_exp', nargs=2, type=int, 
        help='N experiments for each curve (allows plotting 2 curves together).',
        default=(3,3),required=False)
    parser.add_argument('-labels', dest='labels', nargs=2, type=str, 
        help='Curve labels.',
        default=None,required=False)
    
    ##Draw uncertainties
    parser.add_argument('--uncertainty', action='store_true', dest='unc', default=False, 
        help='Plot experiment uncertainties selected acquisitions.')
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int, 
        help='Acquisitions to plot.', default=None,required=False)
    parser.add_argument('-all', action='store_true', dest='all', default=False, 
        help='Plot all acquisitions.')
    parser.add_argument('-ac_func', dest='ac_func', type=str,default='bayesian_bald', 
        help='Function to look for uncertainties.')
    parser.add_argument('-sp', dest='spread', nargs=1, type=int, 
        help='Spread points in interval.', default=10,required=False)
    
    config, unparsed = parser.parse_known_args()

    if config.ids is None and config.sdir is None:
        print("You should either specify an ID or a full path to grab data")
        sys.exit(1)
        
    exp_type = os.path.join(config.results,config.tmode)
    if not os.path.isdir(config.results):
        print("Directory not found: {}".format(config.results))
        sys.exit(1)
        
    if config.multi and not config.stats:
        p = Plotter()
        
        data = p.parseResults(exp_type,config.ids)
        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)        
        p.draw_multiline(data,config.title,config.xtick)
                
    elif config.single:
        p = Plotter(path=config.results)
        p.draw_data(p.parseSlurm(),config.title)

    elif config.unc:
        p = Plotter()

        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)
            
        data = p.retrieveUncertainty(config)
        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)
        p.draw_uncertainty(data,config.ac_n,config.spread,config.title)

    elif config.stats:
        p = Plotter()

        if config.ids is None:
            print("You should define a set of experiment IDs (-id).")
            sys.exit(1)
            
        data = p.parseResults(exp_type,config.ids)
        if config.multi:
            c1 = p.calculate_stats({k:data[k] for k in config.ids[:config.n_exp[0]]},config.auc_only)
            c2 = p.calculate_stats({k:data[k] for k in config.ids[config.n_exp[1]:]},config.auc_only)
            c1.extend(c2)
            data = c1
        else:
            data = p.calculate_stats(data,config.auc_only)
            
        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)
            
        p.draw_stats(data,config.xtick,config.auc_only,config.labels,config.confidence,config.spread,config.title)

        
