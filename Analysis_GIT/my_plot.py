# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:48:52 2017

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:43:47 2016

@author: Alex
"""
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import FigureManagerQT, NavigationToolbar2QT
from PyQt5 import QtWidgets, QtGui

import numpy as np

class myNavigationToolbar2QT(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar2QT.__init__(self, canvas, parent, coordinates=True)
        #self.axes = pl.gca()
        self.myinitdata = 0.
        self.N = 1000
        
        self.Ncontrol= QtWidgets.QLineEdit(self)
        self.Ncontrol.setValidator(QtGui.QIntValidator(self.Ncontrol))
        self.Ncontrol.setFixedWidth(50)
        self.Ncontrol.setText(str(self.N))
        
        self.refr=QtWidgets.QPushButton(QtGui.QIcon('refresh.png'), None, self)
          
        self.refr.clicked.connect(self.getN)
        
        #self.Ncontrol.textChanged.connect(self.getN)
          
        self.Nlabel=QtWidgets.QLabel('Data points on Fig, N=', self)
        
        
        self.Ncact=self.addWidget(self.Nlabel)
        self.Nlact=self.addWidget(self.Ncontrol)
        self.Nrefr=self.addWidget(self.refr)
    
    # to remove matplotlib toolbar status line
    def set_message(self, msg):
        pass
    
    def getN(self):
      self.N = int(self.Ncontrol.text())
      try:
          self.my_rescale()
      except:
          print 'No plot yet to rescale'
    def back(self, *args):
        NavigationToolbar2QT.back(self, *args)
        self.my_rescale()
        
    def forward(self, *args):
        NavigationToolbar2QT.forward(self, *args)
        self.my_rescale()
    
    def release_pan(self, event):
        NavigationToolbar2QT.release_pan(self, event)
        self.my_rescale()
        
    def home(self, *args):
        NavigationToolbar2QT.home(self, *args)
        self.my_rescale()
        
    def release_zoom(self, event):
        NavigationToolbar2QT.release_zoom(self, event)
        self.my_rescale()     
        
    #rescale and replot  
    def my_rescale(self):
        
        t=self.myinitdata[0][0]
        V=self.myinitdata[0][1]
        ax=pl.gca()#self.axes
        (xmin, xmax)=ax.get_xlim()
        rng=xmax-xmin
        #get original data which is in plotting region
        org_points=np.where(((xmin - 2*rng) < t) & (t < (xmax + 2*rng)))
        if (xmin- 2*rng)<t[0] or (xmax+2*rng)>t[-1]:
            k= (-max(xmin-2*rng, t[0])+min(t[-1], xmax+2*rng))/(xmax-xmin)
        else: k=5
        to=t[org_points]
        Vo=V[org_points]
        n=1 #take every nth point from data
        if len(to)/k> self.N:
            n=int(len(to)/k/self.N)
        tcut=to[::n]
        Vcut=Vo[::n]
        
        ax.set_prop_cycle(None)
        ax.clear()
        ax.autoscale(enable=False, axis='x', tight=True)
        ax.set_autoscaley_on(False)
        ax.plot(tcut,Vcut, *self.myinitdata[0][2:], **self.myinitdata[1]) #'.', zorder=0)
        ax.figure.canvas.draw()
        
        

def my_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasQTAgg(figure)
    return myFigureManagerQT(canvas, num)

    


#modified figuremaneger class which uses modified toolbar class
class myFigureManagerQT(FigureManagerQT):
    def __init__(self, canvas, num):
        FigureManagerQT.__init__(self, canvas, num)
        
    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar'] == 'toolbar2':
            toolbar = myNavigationToolbar2QT(canvas, parent, False)
        else:
            toolbar = None
        return toolbar   

def my_plot(*args, **kwargs):
    N=kwargs.pop('N', 10000)
    pl.new_figure_manager = my_figure_manager
    ax = pl.gca()
    # allow callers to override the hold state by passing hold=True|False
    
    #saves input data into toolbar class object and reuses later
    figManager = pl.get_current_fig_manager()
    figManager.toolbar.myinitdata = (args, kwargs)
    figManager.toolbar.N = N
    figManager.toolbar.Ncontrol.setText(str(N))
    #dropping points for first plot
    t=args[0]
    V=args[1]
    rest=args[2:]
    
    n=1 #take every nth point from data
    if len(t)> N:
        n=int(len(t)/N)
    tcut=t[::n]
    Vcut=V[::n]


    try:
        ret = ax.plot(tcut, Vcut, *rest, **kwargs)
        return ret 
    except:
        print "Plotting faild."
    

    
    
        
#for testing        
if __name__ == "__main__":
    my_plot(np.array(range(10000000)),np.sin(np.arange(10000000)/1000000.),'*', color="red", N=100)

    