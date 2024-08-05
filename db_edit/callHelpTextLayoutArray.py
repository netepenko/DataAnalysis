#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:54:16 2024

@author: boeglinw
"""

import sqlite3, sys

from PyQt5.QtWidgets import QDialog, QApplication

import LayoutHelpTextArray as HTA

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
        self.HTA_ui = HTA.Ui_Dialog()
        self.HTA_ui.setupUi(self, self.db_col_data, dialog_size=(450,700))
        self.show_entries()
        self.show()
    
    def show_entries(self):
        # get the current DB entries
        qwhat = '*'
        rows = db.retrieve(self.db_name, qwhat, self.table)
        print(rows)
        if rows != []:
            for i,LH in enumerate(self.HTA_ui.label_Help):
                self.HTA_ui.label_Help[i].setText(str(rows[0][i]))        
            

# for testing
if __name__=="__main__":
    app = QApplication(sys.argv)
    w = TableEntryDialog(30121, 0, 1, 'Rate_Analysis_Help')
    w.show()
    sys.exit(app.exec_())