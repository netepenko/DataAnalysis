#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 17:01:34 2025

@author: boeglinw
"""

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

from PyQt5.QtWidgets import QDialog, QApplication

from testLayoutArrayDialog_ui_edited import Ui_Dialog


# Dialog setup
class testLayoutArrayDialog(qtw.QDialog):
    """ test Dialog """
    def __init__(self, parent = None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = Ui_Dialog()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
        # setup the initial text from a list of lines
        # connections
        self.ui.buttonBox_CancelOK.button(qtw.QDialogButtonBox.Ok).clicked.connect(self.print_ok) 
        self.ui.buttonBox_CancelOK.button(qtw.QDialogButtonBox.Cancel).clicked.connect(self.print_cancel) 

    def print_ok(self):
        # convert the text back to a list of lines once OK has been clicked
        print('button OK was clicked')

    def print_cancel(self):
        # convert the text back to a list of lines once OK has been clicked
        print('button CANCEL was clicked')


# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    h = testLayoutArrayDialog()
    h.show()
    sys.exit(app.exec_())