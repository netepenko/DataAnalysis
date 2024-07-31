#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:07:21 2023

@author: boeglinw
"""

import sys
from PyQt5.QtSql import QSqlDatabase
from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel

# Create the connection
con = QSqlDatabase.addDatabase("QSQLITE")
con.setDatabaseName("../MAST_data/full_shot_listDB.db")

# Create the application
app = QApplication(sys.argv)

# Try to open the connection and handle possible errors
if not con.open():
    QMessageBox.critical(
        None,
        "App Name - Error!",
        "Database Error: %s" % con.lastError().databaseText(),
    )
    sys.exit(1)

# Create the application's window
win = QLabel("Connection Successfully Opened!")
win.setWindowTitle("App Name")
win.resize(200, 100)
win.show()
sys.exit(app.exec_())

