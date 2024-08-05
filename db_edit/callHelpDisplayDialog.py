#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

@author: boeglinw
"""

import sqlite3, sys

from PyQt5.QtWidgets import QDialog, QApplication

import HelpDisplayDialog as HDD

from analysis_modules import database_operations as db


# format dictionary



class HelpDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = HDD.Ui_Dialog()
        self.ui.setupUi(self)
        self.show()
        
# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    h = HelpDialog()
    h.show()
    sys.exit(app.exec_())