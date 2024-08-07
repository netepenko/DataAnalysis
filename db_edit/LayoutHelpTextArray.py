# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'display_helptext_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog, db_data, dialog_size = (415, 711), title = "Variable Help"):
        Dialog.setObjectName("Dialog")
        Dialog.resize(*dialog_size)
        self.title = title
        # setup  dialog window size
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 389, 645))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")


        self.HL = []     # Hor.layout
        self.label_Var = []  # lables
        self.label_Help = []     # line edits
        for i, data in enumerate(db_data):
            var_name = data[1]
            #
            horizontalLayout = QtWidgets.QHBoxLayout()
            horizontalLayout.setObjectName(f"HL_{var_name}_{i}")
            labelVarName = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            labelVarName.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
            labelVarName.setObjectName("label_{var_name}_{i}")
            horizontalLayout.addWidget(labelVarName)
            #
            labelHelpText = QtWidgets.QLabel(self.scrollAreaWidgetContents)
            labelHelpText.setObjectName(f"label_help_{var_name}_{i}")
            # set size policy for help text
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(labelHelpText.sizePolicy().hasHeightForWidth())
            labelHelpText.setSizePolicy(sizePolicy)
            labelHelpText.setMaximumSize(QtCore.QSize(250, 16777215))
            # set alignment
            labelHelpText.setWordWrap(True)
            labelHelpText.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
            horizontalLayout.addWidget(labelHelpText)            
            self.verticalLayout_2.addLayout(horizontalLayout)
            
            # Add elements to list
            self.HL.append(horizontalLayout)
            self.label_Var.append(labelVarName)
            self.label_Help.append(labelHelpText)
        
        
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_1.addWidget(self.scrollArea)
        self.buttonBoxOk = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBoxOk.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBoxOk.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBoxOk.setObjectName("buttonBoxOk")
        self.verticalLayout_1.addWidget(self.buttonBoxOk)

        self.retranslateUi(Dialog, db_data)
        self.buttonBoxOk.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBoxOk.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog, db_data):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", self.title))
        for i, data in enumerate(db_data):
            title_name = data[0]
            help_name = data[1]
            self.label_Var[i].setText(_translate("Dialog", title_name))
            self.label_Help[i].setText(_translate("Dialog", ''))
