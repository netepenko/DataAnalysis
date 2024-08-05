#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

@author: boeglinw
"""

import sqlite3, sys

from PyQt5.QtWidgets import QDialog, QApplication

import LayoutsArray as LA

from analysis_modules import database_operations as db


# format dictionary



class TableEntryDialog(QDialog):
    def __init__(self, shot, channel, version, table):
        super().__init__()
        self.db_name = 'MAST-U_full_shot_listDB_test.db'
        self.table = table
        self.shot = shot
        self.channel = channel
        self.version = version
        # get information about table 
        rows = db.get_table_information(self.db_name, self.table)
        self.db_col_data = []
        self.db_col_types = []
        self.table_col_names = []
        for cid, name, v_type, notnull, dflt_value, pk  in rows:
            self.db_col_data.append((f'Col {cid} : {name} ({v_type}):', f'{cid}_{name}'))
            self.db_col_types.append(v_type)
            self.table_col_names.append(name)
            print(f'{name} has type {v_type}')
        self.LA_ui = LA.Ui_Dialog()
        self.LA_ui.setupUi(self, self.db_col_data, dialog_size=(450,400))
        self.show_entries()
        self.LA_ui.pushButtonSubmit.clicked.connect(self.update_entries)
        self.show()
    
    def show_entries(self):
        # get the current DB entries
        qwhat = '*'
        self.qwhere = f'Shot = {self.shot} and Channel = {self.channel} and Version = {self.version}'
        rows = db.retrieve(self.db_name, qwhat, self.table, where = self.qwhere)
        if rows != []:
            for i,LE in enumerate(self.LA_ui.LE):
                self.LA_ui.LE[i].setText(str(rows[0][i]))        
    
    def update_entries(self):
        table_values = []
        # setore current values in a list
        for i,LE in enumerate(self.LA_ui.LE):
            if self.db_col_types[i] == 'TEXT':
                table_values.append("'" + self.LA_ui.LE[i].text() + "'")
            else:
                table_values.append(self.LA_ui.LE[i].text())
        # update the table
        db.update_row(self.db_name, self.table, self.table_col_names, table_values, self.qwhere)
        

# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = TableEntryDialog(30121, 0, 1, 'Rate_Analysis')
    w.show()
    sys.exit(app.exec_())