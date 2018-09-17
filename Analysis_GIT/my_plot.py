import numpy as np
from types import MethodType
from matplotlib import pyplot as pl
from PyQt5 import QtWidgets, QtGui
   

def back(self, *args):
    self.back_old(*args)
    self.thinning()
    
def forward(self, *args):
    self.forward_old(*args)
    self.thinning()

def release_pan(self, event):
    self.release_pan_old(event)
    self.thinning()
    
def home(self, *args):
    self.home_old(*args)
    self.thinning()
    
def release_zoom(self, event):
    self.release_zoom_old(event)
    self.thinning()     
        
    #rescale and replot
    # this function drops unnecessary ponts in ploting region to reduce
    # plotting delays, but keeps everything above threshold to see peaks
def thinning(self):
    #return
    t=self.t
    V=self.V
    ax=self.axes
    
    (xmin, xmax)=ax.get_xlim()
    rng=xmax-xmin
    #get original data which is in plotting region
    org_points=np.where(((xmin - 1*rng) < t) & (t < (xmax + 1*rng)))
#    if (xmin- 1*rng)<t[0] or (xmax+1*rng)>t[-1]:
#        k= (-max(xmin-1*rng, t[0])+min(t[-1], xmax+1*rng))/(xmax-xmin)
#    else: k=3 # 1 +/-1 regions to the siedes (for dragging thing)
    k=1    
    to=t[org_points]
    Vo=V[org_points]
    n=1 #take every nth point from data
    if len(to)/k> self.N:
        n=int(len(to)/k/self.N)
    tcut=to[::n]
    Vcut=Vo[::n]
    
    ax.set_prop_cycle(None)
    ax.lines.remove(self.axes.lines[0])
    #ax.set_autoscaley_on(True)
    ax.autoscale(enable=False, axis='x', tight=True)
    #ax.autoscale(enable=True, axis='y', tight=True)
    
    ax.plot(tcut,Vcut, *self.rest, **self.kwargs)
    
    ax.figure.canvas.draw()

def getN(self):
  self.N = int(self.Ncontrol.text())
  try:
      self.axes.set_autoscaley_on(True)
      self.thinning()
  except:
      print 'Rescaling did not work for some reason'

def my_plot(*args, **kwargs):
    N=kwargs.pop('N', 10000)
    
    
    #dropping points for first plot
    t=args[0]
    V=args[1]
    rest=args[2:]
    
    n=1 #take every nth point from data
    if len(t)> N:
        n=int(round(len(t)/N))
    tcut=t[::n]
    Vcut=V[::n]
    
    pl.figure()
    a = pl.plot(tcut, Vcut, *rest, **kwargs)
   
    toolbar = a[0].figure.canvas.toolbar
    toolbar.axes = a[0].figure.axes[0]
    
    toolbar.getN = MethodType(getN, toolbar)
    toolbar.thinning = MethodType(thinning, toolbar)
    
    toolbar.back_old = toolbar.back
    toolbar.back = MethodType(back, toolbar)
    toolbar._actions['back'].triggered.disconnect()
    toolbar._actions['back'].triggered.connect(toolbar.back)
    
    toolbar.forward_old = toolbar.forward
    toolbar.forward = MethodType(forward, toolbar)
    toolbar._actions['forward'].triggered.disconnect()
    toolbar._actions['forward'].triggered.connect(toolbar.forward)
    
    toolbar.release_pan_old = toolbar.release_pan
    toolbar.release_pan = MethodType(release_pan, toolbar)
    
    toolbar.home_old = toolbar.home
    toolbar.home = MethodType(home, toolbar)
    toolbar._actions['home'].triggered.disconnect()
    toolbar._actions['home'].triggered.connect(toolbar.home)
    
    toolbar.release_zoom_old = toolbar.release_zoom
    toolbar.release_zoom = MethodType(release_zoom, toolbar)
    
    toolbar.N = N
    toolbar.t = t
    toolbar.V = V
    toolbar.rest =  rest
    toolbar.kwargs = kwargs
    
    toolbar.Ncontrol= QtWidgets.QLineEdit(toolbar)
    toolbar.Ncontrol.setValidator(QtGui.QIntValidator(toolbar.Ncontrol))
    toolbar.Ncontrol.setFixedWidth(50)
    toolbar.Ncontrol.setText(str(N))
    
    
    
    toolbar.refr=QtWidgets.QPushButton(QtGui.QIcon('refresh.png'), None, toolbar)
    toolbar.refr.clicked.connect(toolbar.getN)
    
        
    toolbar.Nlabel=QtWidgets.QLabel('Data points on Fig, N=', toolbar)
    
    toolbar.addWidget(toolbar.Nlabel)
    toolbar.addWidget(toolbar.Ncontrol)
    toolbar.addWidget(toolbar.refr)
    pl.show()

#for testing        
if __name__=='__main__':
    my_plot(np.arange(1000000),np.sin(5*np.pi/1000000*np.arange(1000000)),'.', N=100)



    
    
    
    
 