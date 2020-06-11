#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#import matplotlib
#matplotlib.use('macosx')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import datetime
import numpy as np
import os
import re
import sys
import argparse

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 13}

mpl.rc('font',**font)

palette = plt.get_cmap('Dark2')
newcolors = np.ones((5,4))
newcolors[0,0:3] = np.array([29/256,39/256,125/256])
newcolors[1,0:3] = np.array([110/256,29/256,5/256])
newcolors[2,0:3] = np.array([0/256,5/256,105/256])
newcolors[3,0:3] = np.array([159/256,32/256,54/256])
newcolors[4,0:3] = np.array([76/256,0/256,153/256])
newcolors = np.vstack((palette(np.linspace(0,1,len(palette.colors))),
                           newcolors))
   
palette = ListedColormap(newcolors,name='CustomDark')
del(newcolors)

linestyle = [
    ('solid', (0,())),
    #('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 5))),
    ('densely dotted',        (0, (1, 1))),
    
    #('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),
    
    #('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 3, 1, 3))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),

    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 8, 1, 8, 1, 8))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

#markers = ['','*','+','x','^','.','2','v']
markers = ['*','+','x','^','.','2','v','s','p','D',8,9,10,'4','']

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

    def draw_cluster_distribution(self,data,spread=1.0,title=''):
        """
        Data is returned by parseSlurm. 'clusters' key refers to a dictionary with cluster number as keys. Values are lists:
        [(x,y),(...),...] -> x: number of positive samples; y: number of negative samples
        """
        mpl.rcParams['agg.path.chunksize'] = 1000000
        plots = []
        pl1,pl2 = None,None
        clusters = data['cluster']
        xticks = sorted(list(clusters.keys()))
        selu = lambda x: x if x[0] > x[1] else None
        sell = lambda x: x if x[0] < x[1] else None
        
        #Compute items total
        total = 0
        for c in clusters:
            citems = np.asarray(clusters[c])
            total += np.sum(citems)

        xu,yu = [],[]
        xl,yl = [],[]
        a1,a2 = [],[]
        for k in clusters:
            upper = list(filter(lambda x: not x is None,[selu(x) for x in clusters[k]]))
            lower = list(filter(lambda x: not x is None,[sell(x) for x in clusters[k]]))
            a1.extend([50000*((i[0]+i[1])/total) for i in upper])
            a2.extend([50000*((j[0]+j[1])/total) for j in lower])

            xu.extend(np.random.rand(len(upper))*spread+xticks[k])
            yu.extend([x[0]/(x[0]+x[1]) for x in upper])
            xl.extend(np.random.rand(len(lower))*spread+xticks[k])
            yl.extend([x[1]/(x[0]+x[1]) for x in lower])
            #pl1 = plt.plot(np.random.rand(len(upper))*spread+xticks[k],[x[0]/(x[0]+x[1]) for x in upper],
            #                   'ob',markersize=5,alpha=0.5)
            #pl2 = plt.plot(np.random.rand(len(lower))*spread+xticks[k],[x[1]/(x[0]+x[1]) for x in lower],
            #                   'or',markersize=5,alpha=0.5)
            
        pl1 = plt.scatter(xu,yu,s=a1,c='blue',alpha=0.5,label='Positive')
        pl2 = plt.scatter(xl,yl,s=a2,c='red',alpha=0.5,label='Negative')
            
        #Horizontal line at y=0.5
        plt.plot(xticks,[0.5]*len(xticks),'g--',markersize=3)

        plt.legend(loc=4,ncol=2)
        xticks = list(range(5,len(clusters)+5,5))
        plt.axis([0,xticks[-1]+1,0.4,1.1])
        plt.xticks(xticks)
        plt.yticks(np.arange(0.4, 1.1, 0.2))
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Cluster #")
        plt.ylabel("Percentage")

        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def draw_stats(self,data,xticks,auc_only,labels=None,spread=1,title=''):
        """
        @param data <list>: a list as returned by calculate_stats
        """
        color = 0
        line = 0
        marker = 0
        plots = []
        up = 0.0
        low = 1.0

        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        for d in data:
            x_data,y_data,ci,y_label,color = d
            color = 0 if color < 0 else color
            line = color%len(linestyle)
            marker = color%len(markers)
            color = color % len(palette.colors)
            
            if auc_only and y_label != 'AUC':
                continue
            
            c = plt.plot(x_data, y_data, lw = 2.0, marker=markers[marker],linestyle=linestyle[line][1], color = palette(color), alpha = 1)
            plots.append(c)
            # Shade the confidence interval
            if not np.isnan(ci).any():
                low_ci = np.clip(y_data - ci,0.0,1.0)
                upper_ci = np.clip(y_data + ci,0.0,1.0)
                ym = np.max(upper_ci)
                yl = np.min(low_ci)
            else:
                yl = np.min(y_data)
                ym = np.max(y_data)
            up = ym if ym > up else up
            low = yl if yl < low else low
            if not np.isnan(ci).any():
                plt.fill_between(x_data, low_ci, upper_ci, color = palette(color), alpha = 0.4)
            color += 1
            line = (line+1)%len(linestyle)
            marker = color%len(markers)
            plt.xlabel("Trainset size")
            plt.ylabel(y_label)
                
        # Label the axes and provide a title
        if labels is None:
            labels = ['Mean','{} STD'.format(confidence)]
        plt.legend(plots,labels=labels,loc=4,ncol=2)
        plt.xticks(np.arange(min(x_data), max(x_data)+xticks, xticks))
        ydelta = (1.0 - low)/15
        rg = np.clip(np.arange(max(low,0.0), up if up <= 1.05 else 1.05, ydelta),0.0,1.0)
        np.around(rg,3,rg)
        plt.yticks(rg)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        if max(x_data) > 1000:
            plt.xticks(rotation=30)
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def draw_data(self,data,title='',xlabel='Trainset size',metric=None):

        fig = plt.figure(1)
        fig.suptitle(title)
        fig.subplots_adjust(top=0.5)

        if metric is None:
            tdata = data['time']
            metric = 'time'
        elif metric in data:
            tdata = data[metric]
        else:
            tdata = None
            
        #Train size x Acquisition step time (if that was logged)
        if not tdata is None:
            fig_pos = 211
            if tdata.shape != data['trainset'].shape:
                maxi = min(tdata.shape[0],data['trainset'].shape[0])
            else:
                maxi = max(tdata.shape[0],data['trainset'].shape[0])
        else:
            fig_pos = 110

        if tdata.shape[0] > 0:
            ax = plt.subplot(fig_pos)
            plt.plot(data['trainset'][:maxi],tdata[:maxi],'bo')
            plt.axis([data['trainset'][0]-100,data['trainset'].max()+100,0.0,tdata.max()+.2])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            #ax.yaxis_date()
            formatter = FuncFormatter(self.format_func)
            ax.yaxis.set_major_formatter(formatter)
            #plt.gcf().autofmt_xdate()
            plt.xlabel(xlabel)
            if metric == 'time':
                plt.ylabel('AL step time \n(hh:min:sec)')
            elif metric == 'acqtime':
                plt.ylabel('Acquisition step time \n(hh:min:sec)')
            elif metric == 'traintime':
                plt.ylabel('Training step time \n(hh:min:sec)')
            else:
                plt.ylabel('Time frame not defined')

        #Train size x AUC
        if 'auc' in data and data['auc'].shape[0] > data['trainset'].shape[0]:
            print("AUC results are too many")
            print(data['auc'])
            print(data['trainset'])
            data['auc'] = data['auc'][:-1]

        if 'auc' in data and data['auc'].shape[0] > 0:
            plt.subplot(fig_pos+1)
            min_auc = data['auc'].min()
            plt.plot(data['trainset'],data['auc'],'y-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(xlabel)
            plt.ylabel('AUC')
        elif data['accuracy'].shape[0] > 0:
            plt.subplot(fig_pos+1)
            min_auc = data['accuracy'].min()
            plt.plot(data['trainset'],data['accuracy'],'y-')
            plt.axis([data['trainset'][0],data['trainset'][-1:][0],min_auc-0.1,1.0])
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(xlabel)
            plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def draw_multilabel(self,data,title,xtick,metrics,labels=None):
        lbcount = 0
        color = 0

        if not 'trainset' in data:
            print("Trainset should be provided as X axis")
            sys.exit(1)

        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        fig, ax1 = plt.subplots()
        ax2 = None

        print(data['trainset'])
        for k in metrics:
            if k == 'labels' and data[k].shape[0] > 0:
                #Repeat last point if needed
                if data['trainset'].shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; Labels:{}".format(data['trainset'].shape,data[k].shape))
                    data['labels'] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                yd = [100*(data[k][z][0]/data['trainset'][z]) for z in range(data['trainset'].shape[0])]

                #Prepare plot
                if ax2 is None:
                    ax1.set_ylabel("% Positive",color=palette(color))
                    ax1.plot(data['trainset'],yd, marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax1.set_yticks(np.arange(max(0,min(yd)-10), min(100,10+max(yd)), 5))
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    ax2 = ax1.twinx()
                else:
                    ax2.set_ylabel("% Positive",color=palette(color))
                    ax2.plot(data['trainset'],yd, marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    ax2.set_yticks(np.arange(max(0,min(yd)-10), min(100,10+max(yd)), 5))
                color += 1
                
            elif k == 'auc' and data[k].shape[0] > 0:
                #Repeat last point if needed
                if data['trainset'].shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; AUC:{}".format(data['trainset'].shape,data[k].shape))
                    data[k] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if ax2 is None:
                    ax1.set_ylabel("AUC",color=palette(color))
                    ax1.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    ax1.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    ax2 = ax1.twinx()
                else:
                    ax2.set_ylabel("AUC",color=palette(color))
                    ax2.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=lb)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    ax2.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                color += 1
                
            elif k == 'accuracy' and data[k].shape[0] > 0:
                #Repeat last point if needed
                if data['trainset'].shape[0] > data[k].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; ACC:{}".format(data['trainset'].shape,data[k].shape))
                    data[k] = np.hstack((data[k],data[k][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if ax2 is None:
                    ax1.set_ylabel("Accuracy",color=palette(color))
                    plt.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                    ax1.tick_params(axis='y', labelcolor=palette(color))
                    ax1.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    ax2 = ax1.twinx()
                else:
                    ax2.set_ylabel("Accuracy",color=palette(color))
                    ax2.plot(data['trainset'],data[k], marker='',color=palette(color),linewidth=1,alpha=0.9,label=k)
                    ax2.tick_params(axis='y', labelcolor=palette(color))
                    ax2.set_yticks(np.arange(data[k].min(), 1.0, 0.06))
                    
                color += 1

        fig.legend(bbox_to_anchor=(0.5,0.8),loc=4,ncol=2,labels=config.labels,prop=dict(weight='bold'))
        ax1.set_xticks(np.arange(data['trainset'].min(), data['trainset'].max()+1, xtick))
        if data['trainset'].max() > 1000:
            plt.setp(ax1.get_xticklabels(),rotation=30)
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        ax1.set_xlabel("Training set size")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        
    def draw_multiline(self,data,title,xtick,labels=None,pos=False,auc=False,other=None):
        
        #print("Colors: {}".format(len(palette.colors)))
            
        color = 0
        line = 0
        marker = 0
        plotAUC = False
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        lbcount = 0

        plt.subplots_adjust(left=0.1, right=0.92, bottom=0.19, top=0.92)
        ax = plt.subplot(111)
        for k in data:
            if pos and data[k]['labels'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['labels'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; Labels:{}".format(data[k]['trainset'].shape,data[k]['labels'].shape))
                    data[k]['labels'] = np.hstack((data[k]['labels'],data[k]['labels'][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                yd = [100*(data[k]['labels'][z][0]/data[k]['trainset'][z]) for z in range(data[k]['trainset'].shape[0])]
                if 'color' in data[k]:
                    color = data[k]['color']
                    line = color%len(linestyle)
                    marker = color%len(markers)
                    color = color % len(palette.colors)
                    
                plt.plot(data[k]['trainset'],yd, marker=markers[marker],color=palette(color),linewidth=1.5,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                color += 1
                line = (line+1)%len(linestyle)
                marker = color%len(markers)
                
                plotAUC = False
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(min(yd))
                max_y.append(max(yd))
                print(data[k]['trainset'])                
            elif auc and data[k]['auc'].shape[0] > 0:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['auc'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; AUC:{}".format(data[k]['trainset'].shape,data[k]['auc'].shape))
                    data[k]['auc'] = np.hstack((data[k]['auc'],data[k]['auc'][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if 'color' in data[k]:
                    color = data[k]['color']
                    line = color%len(linestyle)
                    marker = color%len(markers)
                    color = color % len(palette.colors)
                    
                plt.plot(data[k]['trainset'],data[k]['auc'], marker=markers[marker],color=palette(color),linewidth=1.5,linestyle=linestyle[line][1],alpha=0.9,label=lb)
                color += 1
                line = (line+1)%len(linestyle)
                marker = color%len(markers)
                plotAUC = True
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(data[k]['auc'].min())
                max_y.append(data[k]['auc'].max())
                print(data[k]['trainset'])
            elif not other is None:
                if len(other) > 1:
                    print("Multilibe plot only supports a single metric. Using the first one provided ({})".format(other[0]))
                metric = other[0]
                if not metric in data[k]:
                    print("Requested metric not available: {}".format(metric))
                    return None
                else:
                    tdata = data[k][metric]

                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > tdata.shape[0]:
                    print("Shape mismatch:\n Trainset: {}; {}:{}".format(data[k]['trainset'].shape,tdata.shape,metric))
                    tdata = np.hstack((tdata,tdata[-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1
                    
                if 'color' in data[k]:
                    color = data[k]['color']
                    line = color%len(linestyle)
                    marker = color%len(markers)
                    color = color % len(palette.colors)
                    
                plt.plot(data[k]['trainset'],tdata,'bo',marker=markers[marker],linestyle='',color=palette(color),alpha=0.9,label=lb)
                plt.axis([data[k]['trainset'][0]-100,data[k]['trainset'].max()+100,0.0,tdata.max()+.2])
                #fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                #ax.yaxis_date()
                formatter = FuncFormatter(self.format_func)
                ax.yaxis.set_major_formatter(formatter)
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())
                min_y.append(tdata.min())
                max_y.append(tdata.max())
            else:
                #Repeat last point if needed
                if data[k]['trainset'].shape[0] > data[k]['accuracy'].shape[0]:
                    print("Shape mismatch:\n Trainset: {}; ACC:{}".format(data[k]['trainset'].shape,data[k]['accuracy'].shape))
                    data[k]['accuracy'] = np.hstack((data[k]['accuracy'],data[k]['accuracy'][-1:]))

                if labels is None:
                    lb = k
                else:
                    lb = labels[lbcount]
                    lbcount += 1

                if 'color' in data[k]:
                    color = data[k]['color']
                    line = color%len(linestyle)
                    marker = color%len(markers)
                    color = color % len(palette.colors)
                    
                plt.plot(data[k]['trainset'],data[k]['accuracy'], marker=markers[marker],color=palette(color),linewidth=2.0,linestyle=linestyle[line][1],alpha=0.9,label=k)
                color += 1
                line = (line+1)%len(linestyle)
                marker = color%len(markers)
                min_x.append(data[k]['trainset'].min())
                max_x.append(data[k]['trainset'].max())                
                min_y.append(data[k]['accuracy'].min())
                max_y.append(data[k]['accuracy'].max())
                print(data[k]['trainset'])
                
        plt.legend(loc=0,ncol=2,labels=config.labels,prop=dict(weight='bold'))
        plt.xticks(np.arange(min(min_x), max(max_x)+1, xtick))
        if max(max_x) > 1000:
            plt.xticks(rotation=30)
            
        if pos:
            plt.yticks(np.arange(max(0,min(min_y)-10), min(100,10+max(max_y)), 5))
        elif not other is None:
            pass
        else:
            plt.yticks(np.arange(min(0.6,min(min_y)), 1.05, 0.05))
        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Training set size")

        if not other is None:
            metric = other[0]
            if metric == 'time':
                plt.ylabel('AL step time \n(hh:min:sec)')
            elif metric == 'acqtime':
                plt.ylabel('Acquisition step time \n(hh:min:sec)')
            elif metric == 'traintime':
                plt.ylabel('Training step time \n(hh:min:sec)')
        else:
            if plotAUC:
                plt.ylabel("AUC")
            elif pos:
                plt.ylabel("% Positive")
            else:
                plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def plotFromExec(self,data):
        pass

    def parseMetrics(self,data,e_id,metrics):
        split_data = {}
        
        for k in metrics:
            if k == 'pos':
                split_data['labels'] = data['labels']
            else:
                split_data[k] = data[k]
        split_data['trainset'] = data['trainset']

        return split_data

    def parseResults(self,path,al_dirs,n_ids=None,maxx=None):

        def parseDirs(path,al_dirs):
            data = {}
            for d in range(len(al_dirs)):
                if isinstance(path,list):
                    d_path = "{0}-{1}".format(path[d],al_dirs[d])
                else:
                    d_path = "{0}-{1}".format(path,al_dirs[d])
                if os.path.isdir(d_path):
                    data[al_dirs[d]] = self.parseSlurm(d_path,maxx=maxx)
                else:
                    print("Results dir not found: {}".format(d_path))
            return data

        if isinstance(path,list):
            data = {}
            if n_ids is None:
                return parseDirs(path,al_dirs)
            li = 0
            for k in range(len(path)):
                data.update(parseDirs(path[k],al_dirs[li:li+n_ids[k]]))
                li += n_ids[k]
            return data
        else:
            return parseDirs(path,al_dirs)


        
    def format_func(self,x, pos):
        hours = int(x//3600)
        minutes = int((x%3600)//60)
        seconds = int(x%60)

        return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    
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
                              
    def parseSlurm(self,path=None,maxx=None):

        if path is None and self.path is None:
            print("No directory found")
            sys.exit(1)
        elif path is None:
            path = self.path
        elif isinstance(path,list):
            print("Parse a single file at a time")
            return None

        dir_contents = os.listdir(path)
        slurm_path = None
        for fi in dir_contents:
            if fi.startswith('slurm'):
                slurm_path = os.path.join(path,fi)

        if slurm_path is None:
            print("No slurm file in path: {0}".format(path))
            return None
        
        data = {'time':[],
                'acqtime':[],
                'traintime':[],
                'auc':[],
                'trainset':[],
                'accuracy':[],
                'labels':[],
                'cluster':{}}
        start_line = 0
        timerex = r'AL step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        trainrex = r'Training step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        acqrex = r'Acquisition step took: (?P<hours>[0-9]+):(?P<min>[0-9]+):(?P<sec>[0-9]+.[0-9]+)'
        aucrex = r'AUC: (?P<auc>0.[0-9]+)'
        accrex = r'Accuracy: (?P<acc>0.[0-9]+)'
        trainsetrex = r'Train set: (?P<set>[0-9]+) items'
        clusterrex = r'Cluster (?P<cln>[0-9]+) labels: (?P<neg>[0-9]+) are 0; (?P<pos>[0-9]+) are 1;'
        labelrex = r'Train labels: (?P<neg>[0-9]+) are 0; (?P<pos>[0-9]+) are 1;'
        colorrex = r'ColorCode: (?P<code>[0-9]+)'
        
        timerc = re.compile(timerex)
        trainrc = re.compile(trainrex)
        acqrc = re.compile(acqrex)
        aucrc = re.compile(aucrex)
        trainsetrc = re.compile(trainsetrex)
        accrc = re.compile(accrex)
        clusterrc = re.compile(clusterrex)
        labelrc = re.compile(labelrex)
        colorrc = re.compile(colorrex)

        with open(slurm_path,'r') as fd:
            lines = fd.readlines()

        #Set a time reference
        #zero = datetime.datetime(2020,1,1)
        #zero_num = mdates.date2num(zero)
        for line in lines:
            lstrip = line.strip()
            trainmatch = trainrc.fullmatch(lstrip)
            acqmatch = acqrc.fullmatch(lstrip)
            tmatch = timerc.fullmatch(lstrip)
            aucmatch = aucrc.fullmatch(lstrip)
            trmatch = trainsetrc.fullmatch(lstrip)
            accmatch = accrc.fullmatch(lstrip)
            clustermatch = clusterrc.fullmatch(lstrip)
            labelmatch = labelrc.fullmatch(lstrip)
            colormatch = colorrc.fullmatch(lstrip)
            if tmatch:
                td = datetime.timedelta(hours=int(tmatch.group('hours')),minutes=int(tmatch.group('min')),seconds=round(float(tmatch.group('sec'))))
                data['time'].append(td.total_seconds()/3600.0)
            if trainmatch:
                print(lstrip)
                td = datetime.timedelta(hours=int(trainmatch.group('hours')),minutes=int(trainmatch.group('min')),seconds=round(float(trainmatch.group('sec'))))
                data['traintime'].append(td.total_seconds())
            if acqmatch:
                td = datetime.timedelta(hours=int(acqmatch.group('hours')),minutes=int(acqmatch.group('min')),seconds=round(float(acqmatch.group('sec'))))
                data['acqtime'].append(td.total_seconds())
            if aucmatch:
                data['auc'].append(float(aucmatch.group('auc')))
            if trmatch:
                data['trainset'].append(int(trmatch.group('set')))
            if accmatch:
                data['accuracy'].append(float(accmatch.group('acc')))
            if labelmatch:
                data['labels'].append((int(labelmatch.group('pos')),int(labelmatch.group('neg'))))
            if colormatch:
                data['color'] = int(colormatch.group('code'))
                
            if clustermatch:
                cln = int(clustermatch.group('cln'))
                tot = int(clustermatch.group('pos')) + int(clustermatch.group('neg'))
                if tot > 5:
                    if cln in data['cluster']:
                        data['cluster'][cln].append((int(clustermatch.group('pos')),int(clustermatch.group('neg'))))
                    else:
                        data['cluster'][cln] = [(int(clustermatch.group('pos')),int(clustermatch.group('neg')))]
                else:
                    print("Cluster {} has only {} items and will be discarded for plotting".format(cln,tot))
                

        #Use NP arrays
        data['traintime'] = np.asarray(data['traintime'])
        data['acqtime'] = np.asarray(data['acqtime'])
        data['time'] = np.asarray(data['time'])
        data['auc'] = np.asarray(data['auc'])
        data['trainset'] = np.asarray(data['trainset'])
        data['accuracy'] = np.asarray(data['accuracy'])
        data['labels'] = np.asarray(data['labels'])

        if not maxx is None:
            if maxx > np.max(data['trainset']):
                print("Slurm file ({}) does not have that many samples ({}). Maximum is {}.".format(slurm_path,maxx,np.max(data['trainset'])))
                sys.exit(1)
            else:
                upl = np.where(data['trainset'] >= maxx)[0][0]
                upl += 1
            
            data['time'] = data['time'][:upl]
            data['traintime'] = data['traintime'][:upl]
            data['acqtime'] = data['acqtime'][:upl]
            data['auc'] = data['auc'][:upl]
            data['trainset'] = data['trainset'][:upl]
            data['accuracy'] = data['accuracy'][:upl]
            data['labels'] = data['labels'][:upl]

        if data['auc'].shape[0] > 0:
            print("Experiment {}:".format(os.path.basename(path)))
            print("Min AUC: {0}; Max AUC: {1}; Mean AUC: {2:.3f}\n".format(data['auc'].min(),data['auc'].max(),data['auc'].mean()))
        if data['accuracy'].shape[0] > 0:
            print("Experiment {}:".format(os.path.basename(path)))
            print("Min accuracy: {0}; Max accuracy: {1}; Mean accuracy: {2:.3f}\n".format(data['accuracy'].min(),data['accuracy'].max(),data['accuracy'].mean()))

        return data

    def generateDataFromPik(self,pik_dir=None):
        """
        Reads pickle files generated by AL debug function. 
        Returns data to be plotted as if parsed from SLURM files.
        """
        if pik_dir is None:
            pik_dir = self.path
            
        if pik_dir is None or not os.path.isdir(pik_dir):
            return None

        import pickle
        from sklearn import metrics

        def extract_acq(f):
            if f.startswith('al-probs-'):
                return f.split('.')[0].split('-')[3][1:]
            else:
                return -1

        def sort_key(f):
            return int(extract_acq(f))
            
        files = os.listdir(pik_dir)
        files.sort(key=sort_key)
        data = {'accuracy':[],
                'trainset':[]}
        for f in files:
            if f.startswith('al-probs-'):
                fd = open(os.path.join(pik_dir,f),'rb')
                y_true,sprobs = pickle.load(fd)
                fd.close()
                s_pred_all = sprobs[:,:].argmax(axis=0)
                print("Votes array ({})".format(s_pred_all.shape))
                for k in range(0,10):
                    print(s_pred_all[k])
                y_pred = np.asarray([np.bincount(s_pred_all[i]).argmax(axis=0) for i in range(0,s_pred_all.shape[0])])
                print(y_pred)
                acc = metrics.accuracy_score(y_true,y_pred)
                acq = extract_acq(f)
                data['accuracy'].append(acc)
                data['trainset'].append(int(acq))
                print("Acquisition {} accuracy: {}".format(acq,acc))

        data['trainset'] = np.asarray(data['trainset'])
        data['accuracy'] = np.asarray(data['accuracy'])
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

    def calculate_stats(self,data,auc_only,ci):
        """
        @param data <dict>: a dictionary as returned by parseResults

        Calculates mean and standard deviation for AUC and/or Accuracy.

        Returns a list of tuples (trainset,mean_values,std dev,label,color) for each AUC and Accuracy
        """
        auc_value = None
        acc_value = None
        i = 0
        color = -1
        trainset = None
        stats = []

        def calc_ci(val,ci):
            import scipy.stats

            a = np.zeros(shape=val.shape[0],dtype=np.float32)
            for k in range(a.shape[0]):
                n = val[k].shape[0]
                se = scipy.stats.sem(val[k])
                a[k] = se * scipy.stats.t.ppf((1 + ci) / 2., n-1)
            return a

        max_samples = np.inf
        #Check if all experiments had the same number of samples
        for k in data:
            if 'auc' in data[k] and data[k]['auc'].shape[0] > 0:
                max_samples = min(max_samples,len(data[k]['trainset']))
            if not auc_only and 'accuracy' in data[k] and data[k]['accuracy'].shape[0] > 0:
                max_samples = min(max_samples,len(data[k]['trainset']))
                
        for k in data:
            if 'auc' in data[k] and data[k]['auc'].shape[0] > 0:
                if auc_value is None:
                    trainset = data[k]['trainset']
                    shape = (len(data),max_samples)
                    auc_value = np.ndarray(shape=shape,dtype=np.float32)
                #Repeat last point if needed
                if auc_value.shape[1] > data[k]['auc'].shape[0]:
                    print("Experiment {}: repeating last item for AUC data".format(k))
                    data[k]['auc'] = np.concatenate((data[k]['auc'],data[k]['auc'][-1:]),axis=0)
                auc_value[i] = data[k]['auc'][:max_samples]
            if not auc_only and 'accuracy' in data[k] and data[k]['accuracy'].shape[0] > 0:
                if acc_value is None:
                    trainset = data[k]['trainset']
                    shape = (len(data),max_samples)
                    acc_value = np.ndarray(shape=shape,dtype=np.float32)
                #Repeat last point if needed
                if acc_value.shape[1] > data[k]['accuracy'].shape[0]:
                    print("Repeating last item for ACCURACY data")
                    data[k]['accuracy'] = np.concatenate((data[k]['accuracy'],data[k]['accuracy'][-1:]),axis=0)
                acc_value[i] = data[k]['accuracy'][:max_samples]
                
            if 'color' in data[k]:
                color = data[k]['color']
                
            i += 1

        if not auc_value is None:
            stats.append((auc_value,"AUC"))
        if not acc_value is None:
            stats.append((acc_value,"Accuracy"))

        #Return mean and STD dev
        if auc_only:
            return [(trainset[:max_samples],np.mean(auc_value.transpose(),axis=1),calc_ci(auc_value.transpose(),ci),"AUC",color)]
        else:
            return [(trainset[:max_samples],np.mean(arr[0].transpose(),axis=1),np.std(arr[0].transpose(),axis=1),arr[1],color) for arr in stats]
                                                                                                              
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    ##Multiline SLURM parse
    parser.add_argument('--multi', action='store_true', dest='multi', default=False, 
        help='Plot multiple lines from slurm files.')
    parser.add_argument('-pos', action='store_true', dest='pos', default=False, 
        help='Plot percentage of positive patches during aquisition.')    
    parser.add_argument('-ids', dest='ids', nargs='+', type=int, 
        help='Experiment IDs to plot.', default=None,required=False)
    parser.add_argument('-xtick', dest='xtick', nargs=1, type=int, 
        help='xtick interval.', default=200,required=False)
    parser.add_argument('-maxx', dest='maxx', type=int, 
        help='Plot maximum X.', default=None,required=False)
    parser.add_argument('-t', dest='title', type=str,default='AL Experiment', 
        help='Figure title.')
    parser.add_argument('-metrics', dest='metrics', type=str, nargs='+',
        help='Metrics to plot: \n \
        time - AL iteration time; \n \
        traintime - Train step time; \n \
        acqtime - Acquisition step time; \n \
        auc - AUC; \n \
        acc - Accuracy; \n \
        labels - Positive labeled patches acquired,',
       choices=['time','auc','acc','labels','traintime','acqtime'],default=None)    
    parser.add_argument('-type', dest='tmode', type=str, nargs='+',
        help='Experiment type: \n \
        AL - General active learning experiment; \n \
        MN - MNIST dataset experiment.',
       choices=['AL','MN','DB','OR','KM','EN'],default='AL')
    
    ##Single experiment plot
    parser.add_argument('--single', action='store_true', dest='single', default=False, 
        help='Plot data from a single experiment.')
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path (should contain an slurm file).')

    ##Make stats
    parser.add_argument('--stats', action='store_true', dest='stats', default=False, 
        help='Make mean and STD dev from multiple runs.')
    parser.add_argument('-auc', action='store_true', dest='auc_only', default=False, 
        help='Calculate statistics for AUC only.')
    parser.add_argument('-ci', dest='confidence', nargs=1, type=float, 
        help='CI (T-Student).', default=0.95,required=False)
    parser.add_argument('-n', dest='n_exp', nargs='+', type=int, 
        help='N experiments for each curve (allows plotting 2 curves together).',
        default=(3,3),required=False)
    parser.add_argument('-labels', dest='labels', nargs='+', type=str, 
        help='Curve labels.',default=None,required=False)
    
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

    ##Plot debugging data
    parser.add_argument('--debug', action='store_true', dest='debug', default=False, 
        help='Plot experiment uncertainties selected acquisitions.')
    parser.add_argument('-clusters', action='store_true', dest='clusters', default=False, 
        help='Plot cluster composition.')
    
    config, unparsed = parser.parse_known_args()

    if config.ids is None and config.sdir is None:
        print("You should either specify an ID or a full path to grab data")
        sys.exit(1)
        
    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)
        
    if config.multi and not (config.stats or config.debug):
        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)

        if len(config.tmode) == 1:
            exp_type = os.path.join(config.sdir,config.tmode[0],config.tmode[0])
        else:
            exp_type = [os.path.join(config.sdir,tmode,tmode) for tmode in config.tmode]
            
        p = Plotter()

        if not config.metrics is None and len(config.ids) == 1:
            ex_dir = "{}-{}".format(exp_type,str(config.ids[0]))
            data = p.parseMetrics(p.parseSlurm(ex_dir,config.maxx),config.ids[0],config.metrics)
            p.draw_multilabel(data,config.title,config.xtick,config.metrics,config.labels)
        else:
            data = p.parseResults(exp_type,config.ids,maxx=config.maxx)
            if len(data) == 0:
                print("Something is wrong with your command options. No data to plot")
                sys.exit(1)        
            p.draw_multiline(data,config.title,config.xtick,config.labels,config.pos,config.auc_only,config.metrics)
                
    elif config.single:
        p = Plotter(path=config.sdir)
        if not config.metrics is None:
            p.draw_data(p.parseSlurm(),config.title,metric=config.metrics[0])
        else:
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
        if config.sdir is None:
            print("You should specify an experiment directory (use -sd option).")
            sys.exit(1)

        if len(config.tmode) == 1:
            exp_type = os.path.join(config.sdir,config.tmode[0],config.tmode[0])
        else:
            exp_type = []
            for i in range(len(config.n_exp)):
                exp_type.append(os.path.join(config.sdir,config.tmode[i],config.tmode[i]))
        
        if config.ids is None:
            print("You should define a set of experiment IDs (-id).")
            sys.exit(1)

        data = p.parseResults(exp_type,config.ids,config.n_exp,config.maxx)

        if isinstance(config.confidence,list):
            config.confidence = config.confidence[0]

        if config.confidence < 0.0 or config.confidence > 1.0:
            print("CI interval should be between 0.0 and 1.0")
            sys.exit(1)

        if config.multi:
            idx = 0
            c = []
            for i in config.n_exp:
                if i > 0:
                    print("Calculating statistics for experiments {}".format(config.ids[idx:idx+i]))
                    c.extend(p.calculate_stats({k:data[k] for k in config.ids[idx:idx+i]},config.auc_only,config.confidence))
                    idx += i
            data = c
        else:
            data = p.calculate_stats(data,config.auc_only,config.confidence)
            
        if len(data) == 0:
            print("Something is wrong with your command options. No data to plot")
            sys.exit(1)

        p.draw_stats(data,config.xtick,config.auc_only,config.labels,config.spread,config.title)

    elif config.debug:

        if config.sdir is None:
            print("Results dir path is needed (-sd option)")
            sys.exit(1)

        p = Plotter(path=config.sdir)
        if config.clusters:
            data = p.parseSlurm()
        else:
            data = p.generateDataFromPik()
        
        if config.multi:
            #In multi_plot, change the xvalues so that curves reflect the same acquisition
            d2 = p.parseSlurm()
            data['trainset'] = d2['trainset']
            #The debug function only  considers accuracy
            if 'auc' in d2:
                del d2['auc']
            mdata = {'AL selected':data,
                    'AL trained':d2}
            p.draw_multiline(mdata,config.title,config.xtick)
        elif config.clusters:
            #TODO: implement plotting function
            p.draw_cluster_distribution(data,config.spread,config.title)
        else:
            p.draw_data(data,config.title,'Acquisition #')
