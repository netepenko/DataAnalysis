#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:32:26 2024

example of setting up the dialog class associated with the ui.


@author: boeglinw
"""

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

from PyQt5.QtWidgets import QDialog, QApplication

from testDialog_ui import Ui_Dialog


# Dialog setup
class testDialog(qtw.QDialog):
    """ test Dialog """
    def __init__(self, parent = None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = Ui_Dialog()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
        # setup the initial text from a list of lines
        # connections
        self.ui.buttonBox.button(qtw.QDialogButtonBox.Ok).clicked.connect(self.print_ok) 
        self.ui.buttonBox.button(qtw.QDialogButtonBox.Cancel).clicked.connect(self.print_cancel) 

    def print_ok(self):
        # convert the text back to a list of lines once OK has been clicked
        print('button OK was clicked')

    def print_cancel(self):
        # convert the text back to a list of lines once OK has been clicked
        print('button CANCEL was clicked')


# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    h = testDialog()
    h.show()
    sys.exit(app.exec_())