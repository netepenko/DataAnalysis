# -*- coding: utf-8 -*-
#
# Raw data analysis GUI and automatiozation
#
# Created by Alex Netepenko on 6/7/2017
#

from PyQt5 import QtCore, QtWidgets
import channel_data_class as cdc
from rates_plottingclass import rates_plotting as rp
import matplotlib.pyplot as pl


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):

        self.resize(800, 600)
        self.setWindowTitle("Data analysis GUI")

        # create status bar
        self.stBar1 = QtWidgets.QLabel('No file loaded')
        self.stBar1.setFrameStyle(2)
        self.stBar1.setFrameShadow(48)
        self.stBar2 = QtWidgets.QLabel('')
        self.stBar2.setFrameStyle(2)
        self.stBar2.setFrameShadow(48)

        self.statusBar().addWidget(self.stBar1, 1)
        self.statusBar().addWidget(self.stBar2, 1)

        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 780, 560))
        self.tabWidget.setAutoFillBackground(True)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()

        self.tabWidget.addTab(self.tab1, "")
        self.tabWidget.addTab(self.tab2, "")

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1),
                                  "Data analysis")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2),
                                  "Results plotting")

        ###################################
        self.Shot = QtWidgets.QSpinBox(self.tab1)
        self.Shot.setGeometry(QtCore.QRect(40, 20, 150, 22))
        self.Shot.setMinimum(0)
        self.Shot.setMaximum(30000)
        self.Shot.setValue(29975)
        self.Shot.setSingleStep(1)
        self.Shot.setPrefix("Shot: ")

        self.Channel = QtWidgets.QSpinBox(self.tab1)
        self.Channel.setGeometry(QtCore.QRect(200, 20, 150, 22))
        self.Channel.setMinimum(0)
        self.Channel.setMaximum(5)
        self.Channel.setSingleStep(1)
        self.Channel.setPrefix("Channel: ")

        self.dataButton = QtWidgets.QPushButton(self.tab1)
        self.dataButton.setGeometry(QtCore.QRect(370, 19, 100, 23))
        self.dataButton.clicked.connect(self.loadData)
        self.dataButton.setText('Load data')

        self.plotRawButton = QtWidgets.QPushButton(self.tab1)
        self.plotRawButton.setGeometry(QtCore.QRect(490, 19, 100, 23))
        self.plotRawButton.clicked.connect(self.plotRaw)
        self.plotRawButton.setText('Plot data')

        ###################################

        self.chi2 = QtWidgets.QDoubleSpinBox(self.tab1)
        self.chi2.setGeometry(QtCore.QRect(200, 60, 150, 22))
        self.chi2.setMinimum(0.01)
        self.chi2.setMaximum(0.2)
        self.chi2.setValue(0.05)
        self.chi2.setSingleStep(0.01)
        self.chi2.setPrefix("Chi2 = ")

        self.Vgp = QtWidgets.QDoubleSpinBox(self.tab1)
        self.Vgp.setGeometry(QtCore.QRect(40, 60, 150, 22))
        self.Vgp.setMinimum(0.0)
        self.Vgp.setMaximum(2.0)
        self.Vgp.setValue(0.5)
        self.Vgp.setSingleStep(0.1)
        self.Vgp.setPrefix("Vgp = ")

        self.findGood = QtWidgets.QPushButton(self.tab1)
        self.findGood.setGeometry(QtCore.QRect(370, 60, 100, 23))
        self.findGood.clicked.connect(self.findGoodPeaks)
        self.findGood.setText('Find good peaks')

        ####################################

        self.pulsRate = QtWidgets.QDoubleSpinBox(self.tab1)
        self.pulsRate.setGeometry(QtCore.QRect(40, 100, 150, 22))
        self.pulsRate.setMinimum(0.)
        self.pulsRate.setMaximum(1000000.0)
        self.pulsRate.setSingleStep(1000)
        self.pulsRate.setValue(10000.)
        self.pulsRate.setPrefix('Pulser Rate = ')
        self.pulsRate.setSuffix(' Hz')
        self.pulsRate.setDecimals(0)

        self.pulsHight = QtWidgets.QDoubleSpinBox(self.tab1)
        self.pulsHight.setGeometry(QtCore.QRect(200, 100, 150, 22))
        self.pulsHight.setMinimum(0.)
        self.pulsHight.setMaximum(1.5)
        self.pulsHight.setValue(1.2)
        self.pulsHight.setSingleStep(0.1)
        self.pulsHight.setPrefix('Pulser hight = ')
        self.pulsHight.setSuffix(' V')
        self.pulsHight.setDecimals(1)

        self.addPulser = QtWidgets.QPushButton(self.tab1)
        self.addPulser.setGeometry(QtCore.QRect(370, 100, 100, 23))
        self.addPulser.clicked.connect(self.addPuls)
        self.addPulser.setText('Add pulser')

        #####################################

        self.intb = QtWidgets.QDoubleSpinBox(self.tab1)
        self.intb.setGeometry(QtCore.QRect(40, 140, 150, 22))
        self.intb.setMinimum(0.)
        self.intb.setValue(0.)
        self.intb.setDecimals(4)
        self.intb.setSingleStep(0.1)
        self.intb.setPrefix("Interval to fit = ")

        self.inte = QtWidgets.QDoubleSpinBox(self.tab1)
        self.inte.setGeometry(QtCore.QRect(200, 140, 150, 22))
        self.inte.setMinimum(0.0)
        self.inte.setDecimals(4)
        self.inte.setValue(0.1)
        self.inte.setSingleStep(0.01)
        self.inte.setSuffix(" s")

        self.fitInt = QtWidgets.QPushButton(self.tab1)
        self.fitInt.setGeometry(QtCore.QRect(370, 140, 100, 23))
        self.fitInt.clicked.connect(self.fitInterval)
        self.fitInt.setText('Fit interval')

        self.plfit = QtWidgets.QCheckBox(self.tab1)
        self.plfit.setGeometry(QtCore.QRect(500, 144, 15, 15))
        self.plfit.setChecked(False)

        self.plLaber = QtWidgets.QLabel(self.tab1)
        self.plLaber.setGeometry(QtCore.QRect(520, 140, 100, 23))
        self.plLaber.setText('Plot fitting results')

        ################################################

        self.Shot1 = QtWidgets.QSpinBox(self.tab2)
        self.Shot1.setGeometry(QtCore.QRect(40, 20, 150, 22))
        self.Shot1.setMinimum(0)
        self.Shot1.setMaximum(30000)
        self.Shot1.setValue(29975)
        self.Shot1.setSingleStep(1)
        self.Shot1.setPrefix("Shot: ")

        self.Channel1 = QtWidgets.QSpinBox(self.tab2)
        self.Channel1.setGeometry(QtCore.QRect(200, 20, 150, 22))
        self.Channel1.setMinimum(0)
        self.Channel1.setMaximum(5)
        self.Channel1.setSingleStep(1)
        self.Channel1.setPrefix("Channel: ")

        self.plFsel = QtWidgets.QPushButton(self.tab2)
        self.plFsel.setGeometry(QtCore.QRect(370, 20, 100, 23))
        self.plFsel.clicked.connect(self.selectplFile)
        self.plFsel.setText('Select fit results')

        #######################################################################
        self.tsl = QtWidgets.QDoubleSpinBox(self.tab2)
        self.tsl.setGeometry(QtCore.QRect(40, 60, 150, 22))
        self.tsl.setMinimum(0.0001)
        self.tsl.setDecimals(4)
        self.tsl.setValue(0.001)
        self.tsl.setSingleStep(0.001)
        self.tsl.setPrefix("Time slice width = ")
        self.tsl.setSuffix(" s")

        self.plotRateBut = QtWidgets.QPushButton(self.tab2)
        self.plotRateBut.setGeometry(QtCore.QRect(370, 60, 100, 23))
        self.plotRateBut.clicked.connect(self.plotRate)
        self.plotRateBut.setText('Plot rate')

        self.bins = QtWidgets.QSpinBox(self.tab2)
        self.bins.setGeometry(QtCore.QRect(40, 100, 150, 22))
        self.bins.setMinimum(10)
        self.bins.setMaximum(500)
        self.bins.setValue(150)
        self.bins.setSingleStep(5)
        self.bins.setPrefix("Bins = ")

        self.sigRat = QtWidgets.QDoubleSpinBox(self.tab2)
        self.sigRat.setGeometry(QtCore.QRect(40, 140, 150, 22))
        self.sigRat.setMinimum(0.0)
        self.sigRat.setMaximum(500)
        self.sigRat.setValue(100)
        self.sigRat.setSingleStep(10)
        self.sigRat.setPrefix("Sig_ratio = ")

        self.cumulHist = QtWidgets.QPushButton(self.tab2)
        self.cumulHist.setGeometry(QtCore.QRect(370, 100, 100, 23))
        self.cumulHist.clicked.connect(self.cumulativHisto)
        self.cumulHist.setText('Cumulative histo')

        self.tabWidget.setCurrentIndex(0)

    def selectplFile(self):
        self.fileDialog = QtWidgets.QFileDialog(self)
        self.fileDialog.setDirectory('../Analysis_Results/%d/Raw_Fitting/' %
                                     self.Shot1.value())
        self.plFile = self.fileDialog.getOpenFileName()
        self.fileDialog.destroy()

    def plotRate(self):
        plov = rp(self.Shot1.value(), self.Channel1.value(), self.plFile[0])
        plov.par['sig_ratio'] = self.sigRat.value()
        plov.par['time_slice_width'] = self.tsl.value()
        plov.plot_results()

    def addPuls(self):
        self.data.par['P_amp'] = self.pulsHight.value()
        self.data.par['pulser_rate'] = self.pulsRate.value()
        self.data.add_pulser()

    def findGoodPeaks(self):
        self.data.chi2 = self.chi2.value()
        self.data.Vgp = self.Vgp.value()
        self.data.peak_shape()

    def fitInterval(self):
        self.data.fit_interval(self.intb.value(), self.inte.value(),
                               self.plfit.checkState())

    def loadData(self):
        try:
            self.data = cdc.channel_data(self.Shot.value(),
                                         self.Channel.value())
            self.stBar1.setText(self.data.par['exp_file'])
            self.intb.setValue(self.data.par['dtmin']/10**6)
            self.inte.setValue(self.data.par['dtmax']/10**6)
        except:
            self.stBar1.setText("Couldn't load the data")

#    def selectFile(self):
#
#        self.fileDialog=QtWidgets.QFileDialog(self.centralwidget)
#        self.fileDialog.setDirectory('../Raw_Data')
#        self.dataFile=self.fileDialog.getOpenFileName()
#        self.fileDialog.destroy()
#        if self.dataFile[0] != '':
#                self.dataDisp.setText(self.dataFile[0].rsplit('/',1)[-1])
#                self.dataFile=self.dataFile[0]
#        else:
#            self.dataFile=''

    def cumulativHisto(self):
        try:
            plov = rp(self.Shot.value(), self.Channel.value(), self.plFile[0])
            plov.par['sig_ratio'] = self.sigRat.value()
            plov.par['h_bins'] = self.bins.value()
            plov.plot_results()
            pl.close()
            plov.cumulative_hist()
        except:
            self.errormsg('Plot rates first')

    def plotRaw(self):
        try:
            self.data.plot_raw()
        except:
            self.errormsg('Load data first')

    def errormsg(self, text):
        QtWidgets.QMessageBox.warning(self, 'Message', text)

if __name__ == "__main__":
    import sys
    # check if Qt app exists already
    app = QtCore.QCoreApplication.instance()
    if app is None:
        # create one if no
        app = QtWidgets.QApplication(sys.argv)

    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()

    sys.exit(app.exec_())
