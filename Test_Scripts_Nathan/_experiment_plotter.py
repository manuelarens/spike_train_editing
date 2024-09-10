# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plotter.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from pyqtgraph import GraphicsLayoutWidget


class Ui_Experiment_Plotter(object):
    def setupUi(self, Plotter):
        if not Plotter.objectName():
            Plotter.setObjectName(u"Plotter")
        Plotter.resize(757, 490)
        self.gridLayout = QGridLayout(Plotter)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_sidebar = QFrame(Plotter)
        self.frame_sidebar.setObjectName(u"frame_sidebar")
        self.frame_sidebar.setMinimumSize(QSize(300, 0))
        self.frame_sidebar.setMaximumSize(QSize(300, 16777215))
        self.frame_sidebar.setFrameShape(QFrame.StyledPanel)
        self.frame_sidebar.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_sidebar)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_sidebar_title = QLabel(self.frame_sidebar)
        self.lbl_sidebar_title.setObjectName(u"lbl_sidebar_title")
        self.lbl_sidebar_title.setMaximumSize(QSize(16777215, 0))
        font = QFont()
        font.setPointSize(12)
        self.lbl_sidebar_title.setFont(font)
        self.lbl_sidebar_title.setAlignment(Qt.AlignCenter)
        self.lbl_sidebar_title.setWordWrap(True)

        self.verticalLayout.addWidget(self.lbl_sidebar_title)

        self.btn_freeze = QPushButton(self.frame_sidebar)
        self.btn_freeze.setObjectName(u"btn_freeze")
        self.btn_freeze.setMaximumSize(QSize(10000000, 16777215))

        self.verticalLayout.addWidget(self.btn_freeze)

        self.group_channels = QGroupBox(self.frame_sidebar)
        self.group_channels.setObjectName(u"group_channels")
        self.gridLayout_2 = QGridLayout(self.group_channels)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.btn_enable_all_channels = QPushButton(self.group_channels)
        self.btn_enable_all_channels.setObjectName(u"btn_enable_all_channels")

        self.gridLayout_2.addWidget(self.btn_enable_all_channels, 0, 0, 1, 1)

        self.btn_disable_all_channels = QPushButton(self.group_channels)
        self.btn_disable_all_channels.setObjectName(u"btn_disable_all_channels")

        self.gridLayout_2.addWidget(self.btn_disable_all_channels, 0, 1, 1, 1)

        self.btn_autoscale = QPushButton(self.group_channels)
        self.btn_autoscale.setObjectName(u"btn_autoscale")

        self.gridLayout_2.addWidget(self.btn_autoscale, 1, 0, 1, 2)


        self.verticalLayout.addWidget(self.group_channels)

        self.group_amplitude = QGroupBox(self.frame_sidebar)
        self.group_amplitude.setObjectName(u"group_amplitude")
        self.gridLayout_3 = QGridLayout(self.group_amplitude)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.lbl_amplitude_UNI = QLabel(self.group_amplitude)
        self.lbl_amplitude_UNI.setObjectName(u"lbl_amplitude_UNI")

        self.gridLayout_3.addWidget(self.lbl_amplitude_UNI, 0, 0, 1, 1)

        self.spin_amplitude = QSpinBox(self.group_amplitude)
        self.spin_amplitude.setObjectName(u"spin_amplitude")
        self.spin_amplitude.setAlignment(Qt.AlignCenter)
        self.spin_amplitude.setMaximum(100000)

        self.gridLayout_3.addWidget(self.spin_amplitude, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.group_amplitude)
    	
        # determine offset box
        self.calibration_box = QGroupBox(self.frame_sidebar)
        self.calibration_box.setObjectName(u"calibration_box")
        self.gridLayout_7 = QGridLayout(self.calibration_box)
        self.gridLayout_7.setObjectName(u"gridLayout_7")


        self.btn_calibration = QPushButton(self.calibration_box)
        self.btn_calibration.setObjectName(u"btn_calibration")
        self.gridLayout_7.addWidget(self.btn_calibration, 0, 0, 1, 2)
        

        self.lbl_calibration_timer = QLabel(self.calibration_box)
        self.lbl_calibration_timer.setObjectName(u"lbl_calibration_timer")
        self.calibration_seconds = 5
        self.offset_timer = QTimer()
        self.offset_timer.timeout.connect(lambda: self.update_offset_label())

        self.gridLayout_7.addWidget(self.lbl_calibration_timer, 1, 0 ,1, -1)
        self.lbl_calibration_timer.setText(f'{self.calibration_seconds}')
        self.lbl_calibration_timer.setStyleSheet("font-size: 32px;")
        self.lbl_calibration_timer.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.calibration_box)


        ## begin of MVC box
        self.mvc_buttons = QGroupBox(self.frame_sidebar)
        self.mvc_buttons.setObjectName(u"mvc_buttons")
        self.gridLayout_4 = QGridLayout(self.mvc_buttons)
        self.gridLayout_4.setObjectName(u"gridLayout_4")

        # duration settings MVC
        self.lbl_mvc_duration = QLabel(self.mvc_buttons)
        self.lbl_mvc_duration.setObjectName(u"lbl_mvc_duration")
        self.gridLayout_4.addWidget(self.lbl_mvc_duration, 0, 0, 1, 1)

        self.mvc_duration = QSpinBox(self.mvc_buttons)
        self.mvc_duration.setObjectName(u"mvc_duration")
        self.mvc_duration.setAlignment(Qt.AlignCenter)
        self.mvc_duration.setMaximum(100000)
        self.mvc_duration.setValue(3)
        self.gridLayout_4.addWidget(self.mvc_duration, 0, 1, 1, 1)

        # rest settings MVC
        self.lbl_mvc_rest = QLabel(self.mvc_buttons)
        self.lbl_mvc_rest.setObjectName(u"lbl_mvc_rest")
        self.gridLayout_4.addWidget(self.lbl_mvc_rest, 1, 0, 1, 1)

        self.mvc_rest = QSpinBox(self.mvc_buttons)
        self.mvc_rest.setObjectName(u"mvc_rest")
        self.mvc_rest.setAlignment(Qt.AlignCenter)
        self.mvc_rest.setMaximum(100000)
        self.mvc_rest.setValue(30)
        self.gridLayout_4.addWidget(self.mvc_rest, 1, 1, 1, 1)

        # repeat settings MVC
        self.lbl_mvc_repeats = QLabel(self.mvc_buttons)
        self.lbl_mvc_repeats.setObjectName(u"lbl_mvc_repeats")
        self.gridLayout_4.addWidget(self.lbl_mvc_repeats, 2, 0, 1, 1)
        
        self.mvc_repeats = QSpinBox(self.mvc_buttons)
        self.mvc_repeats.setObjectName(u"mvc_repeats")
        self.mvc_repeats.setAlignment(Qt.AlignCenter)
        self.mvc_repeats.setMaximum(100000)
        self.mvc_repeats.setValue(3)
        self.gridLayout_4.addWidget(self.mvc_repeats, 2, 1, 1, 1)

        self.start_rest = 10
        self.lbl_timer_value = QLabel(self.mvc_buttons)
        self.lbl_timer_value.setObjectName(u"lbl_timer_value")
        self.lbl_timer_value.setText(f"{self.start_rest}")

        self.gridLayout_4.addWidget(self.lbl_timer_value, 3, 0 ,1, -1)
        self.lbl_timer_value.setStyleSheet("font-size: 32px;")
        self.lbl_timer_value.setAlignment(Qt.AlignCenter)

        self.btn_start_MVC = QPushButton(self.mvc_buttons)
        self.btn_start_MVC.setObjectName(u"btn_start_MVC")
        self.gridLayout_4.addWidget(self.btn_start_MVC, 4, 0, 1, 2)

        self.verticalLayout.addWidget(self.mvc_buttons)

        ## begin of the Experiment box
        self.training_buttons = QGroupBox(self.frame_sidebar)
        self.training_buttons.setObjectName(u"training_buttons")
        self.gridLayout_5 = QGridLayout(self.training_buttons)
        self.gridLayout_5.setObjectName(u"gridLayout_5")

        #Training slope
        self.lbl_slope_value = QLabel(self.training_buttons)
        self.lbl_slope_value.setObjectName(u"lbl_mvc_slope")
        self.gridLayout_5.addWidget(self.lbl_slope_value, 1, 0, 1, 1)

        self.slope_value = QSpinBox(self.training_buttons)
        self.slope_value.setObjectName(u"mvc_slope")
        self.slope_value.setAlignment(Qt.AlignCenter)
        self.slope_value.setMaximum(100000)
        self.slope_value.setValue(5)
        self.gridLayout_5.addWidget(self.slope_value, 1, 1 ,1, 1)

        #Training level
        self.lbl_level_value = QLabel(self.training_buttons)
        self.lbl_level_value.setObjectName(u"lbl_mvc_level")
        self.gridLayout_5.addWidget(self.lbl_level_value, 2, 0, 1, 1)

        self.level_value = QSpinBox(self.training_buttons)
        self.level_value.setObjectName(u"lbl_mvc_level")
        self.level_value.setAlignment(Qt.AlignCenter)
        self.level_value.setMaximum(100000)
        self.level_value.setValue(30)
        self.gridLayout_5.addWidget(self.level_value, 2, 1, 1, 1)


        #Training duration
        self.lbl_level_duration_value = QLabel(self.training_buttons)
        self.lbl_level_duration_value.setObjectName(u"lbl_level_duration")
        self.gridLayout_5.addWidget(self.lbl_level_duration_value, 3, 0, 1, 1)

        self.level_duration_value = QSpinBox(self.training_buttons)
        self.level_duration_value.setObjectName(u"level_duration")
        self.level_duration_value.setAlignment(Qt.AlignCenter)
        self.level_duration_value.setMaximum(100000)
        self.level_duration_value.setValue(30)
        self.gridLayout_5.addWidget(self.level_duration_value, 3, 1, 1, 1)
        
        # Training rest
        self.lbl_rest_value = QLabel(self.training_buttons)
        self.lbl_rest_value.setObjectName(u"lbl_rest")
        self.gridLayout_5.addWidget(self.lbl_rest_value, 4, 0, 1, 1)

        self.rest_value = QSpinBox(self.training_buttons)
        self.rest_value.setObjectName(u"rest")
        self.rest_value.setAlignment(Qt.AlignCenter)
        self.rest_value.setMaximum(100000)
        self.rest_value.setValue(30)
        self.gridLayout_5.addWidget(self.rest_value, 4, 1, 1, 1)

        # Training repeats
        self.lbl_repeat_value = QLabel(self.training_buttons)
        self.lbl_repeat_value.setObjectName(u"lbl_repeats")
        self.gridLayout_5.addWidget(self.lbl_repeat_value, 5, 0, 1, 1)

        self.repeat_value = QSpinBox(self.training_buttons)
        self.repeat_value.setObjectName(u"repeats")
        self.repeat_value.setAlignment(Qt.AlignCenter)
        self.repeat_value.setMaximum(100000)
        self.repeat_value.setValue(1)
        self.gridLayout_5.addWidget(self.repeat_value, 5, 1, 1, 1)

        # y axis resolution
        self.lbl_y_resolution = QLabel(self.training_buttons)
        self.lbl_y_resolution.setObjectName(u"lbl_repeats")
        self.gridLayout_5.addWidget(self.lbl_y_resolution, 6, 0, 1, 1)

        self.y_resolution_value = QSpinBox(self.training_buttons)
        self.y_resolution_value.setObjectName(u"repeats")
        self.y_resolution_value.setAlignment(Qt.AlignCenter)
        self.y_resolution_value.setMaximum(100000)
        self.y_resolution_value.setValue(10)
        self.gridLayout_5.addWidget(self.y_resolution_value, 6, 1, 1, 1)


        #Start measurement button
        self.btn_preview = QPushButton(self.training_buttons)
        self.btn_preview.setObjectName(u"btn_preview")

        self.gridLayout_5.addWidget(self.btn_preview, 7, 0, 1, 1)

        self.btn_start_measurement = QPushButton(self.training_buttons)
        self.btn_start_measurement.setObjectName(u"btn_start_measurement")

        self.gridLayout_5.addWidget(self.btn_start_measurement, 7, 1, 1, 1)

        self.verticalLayout.addWidget(self.training_buttons)


        ## begin of the Live box
        self.live = QGroupBox(self.frame_sidebar)
        self.live.setObjectName(u"live")
        self.gridLayout_8 = QGridLayout(self.live)
        self.gridLayout_8.setObjectName(u"gridLayout_8")

        #Training slope
        self.lbl_live_slope_value = QLabel(self.live)
        self.lbl_live_slope_value.setObjectName(u"lbl_mvc_slope")
        self.gridLayout_8.addWidget(self.lbl_live_slope_value, 1, 0, 1, 1)

        self.live_slope_value = QSpinBox(self.live)
        self.live_slope_value.setObjectName(u"mvc_slope")
        self.live_slope_value.setAlignment(Qt.AlignCenter)
        self.live_slope_value.setMaximum(100000)
        self.live_slope_value.setValue(5)
        self.gridLayout_8.addWidget(self.live_slope_value, 1, 1 ,1, 1)

        #Training level
        self.lbl_live_level_value = QLabel(self.live)
        self.lbl_live_level_value.setObjectName(u"lbl_mvc_level")
        self.gridLayout_8.addWidget(self.lbl_live_level_value, 2, 0, 1, 1)

        self.live_level_value = QSpinBox(self.live)
        self.live_level_value.setObjectName(u"lbl_mvc_level")
        self.live_level_value.setAlignment(Qt.AlignCenter)
        self.live_level_value.setMaximum(100000)
        self.live_level_value.setValue(30)
        self.gridLayout_8.addWidget(self.live_level_value, 2, 1, 1, 1)


        #Training duration
        self.lbl_live_level_duration_value = QLabel(self.live)
        self.lbl_live_level_duration_value.setObjectName(u"lbl_level_duration")
        self.gridLayout_8.addWidget(self.lbl_live_level_duration_value, 3, 0, 1, 1)

        self.live_level_duration_value = QSpinBox(self.live)
        self.live_level_duration_value.setObjectName(u"level_duration")
        self.live_level_duration_value.setAlignment(Qt.AlignCenter)
        self.live_level_duration_value.setMaximum(100000)
        self.live_level_duration_value.setValue(30)
        self.gridLayout_8.addWidget(self.live_level_duration_value, 3, 1, 1, 1)
        
        # Training rest
        self.lbl_live_rest_value = QLabel(self.live)
        self.lbl_live_rest_value.setObjectName(u"lbl_rest")
        self.gridLayout_8.addWidget(self.lbl_live_rest_value, 4, 0, 1, 1)

        self.live_rest_value = QSpinBox(self.live)
        self.live_rest_value.setObjectName(u"rest")
        self.live_rest_value.setAlignment(Qt.AlignCenter)
        self.live_rest_value.setMaximum(100000)
        self.live_rest_value.setValue(30)
        self.gridLayout_8.addWidget(self.live_rest_value, 4, 1, 1, 1)

        # Training repeats
        self.lbl_live_repeat_value = QLabel(self.live)
        self.lbl_live_repeat_value.setObjectName(u"lbl_repeats")
        self.gridLayout_8.addWidget(self.lbl_live_repeat_value, 5, 0, 1, 1)

        self.live_repeat_value = QSpinBox(self.live)
        self.live_repeat_value.setObjectName(u"repeats")
        self.live_repeat_value.setAlignment(Qt.AlignCenter)
        self.live_repeat_value.setMaximum(100000)
        self.live_repeat_value.setValue(1)
        self.gridLayout_8.addWidget(self.live_repeat_value, 5, 1, 1, 1)

        # y axis resolution
        self.lbl_live_y_resolution = QLabel(self.live)
        self.lbl_live_y_resolution.setObjectName(u"lbl_repeats")
        self.gridLayout_8.addWidget(self.lbl_live_y_resolution, 6, 0, 1, 1)

        self.live_y_resolution_value = QSpinBox(self.live)
        self.live_y_resolution_value.setObjectName(u"repeats")
        self.live_y_resolution_value.setAlignment(Qt.AlignCenter)
        self.live_y_resolution_value.setMaximum(100000)
        self.live_y_resolution_value.setValue(10)
        self.gridLayout_8.addWidget(self.live_y_resolution_value, 6, 1, 1, 1)



        #Start measurement button
        self.live_btn_preview = QPushButton(self.live)
        self.live_btn_preview.setObjectName(u"live_btn_preview")

        self.gridLayout_8.addWidget(self.live_btn_preview, 7, 0, 1, 1)

        self.live_btn_start_measurement = QPushButton(self.live)
        self.live_btn_start_measurement.setObjectName(u"btn_start_live")

        self.gridLayout_8.addWidget(self.live_btn_start_measurement, 7, 1, 1, 1)

        self.verticalLayout.addWidget(self.live)


        self.scrollArea = QScrollArea(self.frame_sidebar)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 296, 202))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.layout_channels = QVBoxLayout()
        self.layout_channels.setObjectName(u"layout_channels")

        self.verticalLayout_3.addLayout(self.layout_channels)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout.addWidget(self.scrollArea)

        self.frame_logo = QFrame(self.frame_sidebar)
        self.frame_logo.setObjectName(u"frame_logo")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_logo.sizePolicy().hasHeightForWidth())
        self.frame_logo.setSizePolicy(sizePolicy)
        self.frame_logo.setMaximumSize(QSize(16777215, 80))
        self.frame_logo.setFrameShape(QFrame.StyledPanel)
        self.frame_logo.setFrameShadow(QFrame.Raised)
        self.gridLayout_6 = QGridLayout(self.frame_logo)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setSizeConstraint(QLayout.SetFixedSize)
        self.gridLayout_6.setVerticalSpacing(0)
        self.gridLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.horizontalSpacer_logo1 = QSpacerItem(68, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_logo1, 0, 0, 1, 1)

        self.horizontalSpacer_logo2 = QSpacerItem(70, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_logo2, 0, 2, 1, 1)

        self.label_logo = QLabel(self.frame_logo)
        self.label_logo.setObjectName(u"label_logo")
        sizePolicy.setHeightForWidth(self.label_logo.sizePolicy().hasHeightForWidth())
        self.label_logo.setSizePolicy(sizePolicy)
        self.label_logo.setMinimumSize(QSize(143, 62))
        self.label_logo.setMaximumSize(QSize(143, 62))
        self.label_logo.setCursor(QCursor(Qt.ArrowCursor))
        self.label_logo.setLayoutDirection(Qt.LeftToRight)
        self.label_logo.setFrameShape(QFrame.NoFrame)
        self.label_logo.setFrameShadow(QFrame.Plain)
        self.label_logo.setPixmap(QPixmap(u"../../Media/Images/TMSi_logo.PNG"))
        self.label_logo.setScaledContents(True)
        self.label_logo.setAlignment(Qt.AlignCenter)
        self.label_logo.setMargin(0)

        self.gridLayout_6.addWidget(self.label_logo, 0, 1, 1, 1)


        self.verticalLayout.addWidget(self.frame_logo)

        self.btn_hide_sidebar = QPushButton(self.frame_sidebar)
        self.btn_hide_sidebar.setObjectName(u"btn_hide_sidebar")

        self.verticalLayout.addWidget(self.btn_hide_sidebar)


        self.gridLayout.addWidget(self.frame_sidebar, 0, 0, 1, 1)

        self.frame_chart = QFrame(Plotter)
        self.frame_chart.setObjectName(u"frame_chart")
        self.frame_chart.setFrameShape(QFrame.StyledPanel)
        self.frame_chart.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_chart)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.chart = GraphicsLayoutWidget(self.frame_chart)
        self.chart.setObjectName(u"chart")

        self.horizontalLayout_3.addWidget(self.chart)

        self.legend = GraphicsLayoutWidget(self.frame_chart)
        self.legend.setObjectName(u"legend")
        self.legend.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout_3.addWidget(self.legend)


        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)


        self.gridLayout.addWidget(self.frame_chart, 0, 1, 1, 1)


        self.retranslateUi(Plotter)

        QMetaObject.connectSlotsByName(Plotter)
    # setupUi

    def retranslateUi(self, Plotter):
        Plotter.setWindowTitle(QCoreApplication.translate("Plotter", u"Form", None))
        self.lbl_sidebar_title.setText(QCoreApplication.translate("Plotter", u"PLOTTER", None))
        self.btn_freeze.setText(QCoreApplication.translate("Plotter", u"Pause Viewer", None))
        self.group_channels.setTitle(QCoreApplication.translate("Plotter", u"Channels", None))
        self.btn_enable_all_channels.setText(QCoreApplication.translate("Plotter", u"Enable all", None))
        self.btn_disable_all_channels.setText(QCoreApplication.translate("Plotter", u"Disable all", None))
        self.btn_autoscale.setText(QCoreApplication.translate("Plotter", u"Autoscale", None))
        self.calibration_box.setTitle(QCoreApplication.translate("Plotter", u"Calibration", None))
        self.btn_calibration.setText(QCoreApplication.translate("Plotter", u"Start Procedure", None))
        self.mvc_buttons.setTitle(QCoreApplication.translate("Plotter", u"MVC", None))
        self.lbl_mvc_duration.setText(QCoreApplication.translate("Plotter", u"MVC duration (s)", None))
        self.lbl_mvc_rest.setText(QCoreApplication.translate("Plotter", u"MVC rest (s)", None))
        self.lbl_mvc_repeats.setText(QCoreApplication.translate("Plotter", u"MVC repeats (#)", None))
        self.btn_start_MVC.setText(QCoreApplication.translate("Plotter", u"Start MVC", None))
        self.training_buttons.setTitle(QCoreApplication.translate("Plotter", u"Training", None))
        self.lbl_slope_value.setText(QCoreApplication.translate("Plotter", u"Slope (%MVC/s)", None))
        self.lbl_level_value.setText(QCoreApplication.translate("Plotter", u"Level (% MVC)", None))
        self.lbl_level_duration_value.setText(QCoreApplication.translate("Plotter", u"Duration (s)", None))
        self.lbl_rest_value.setText(QCoreApplication.translate("Plotter", u"Rest (s)", None))
        self.lbl_repeat_value.setText(QCoreApplication.translate("Plotter", u"Repeats (#)", None))
        self.lbl_y_resolution.setText(QCoreApplication.translate("Plotter", u"Y Resolution (% MVC)", None))
        self.btn_preview.setText(QCoreApplication.translate("Plotter", u"Preview Profile", None))
        self.btn_start_measurement.setText(QCoreApplication.translate("Plotter", u"Start Experiment", None))
        self.live.setTitle(QCoreApplication.translate("Plotter", u"Live Decomposition", None))
        self.lbl_live_slope_value.setText(QCoreApplication.translate("Plotter", u"Slope (%MVC/s)", None))
        self.lbl_live_level_value.setText(QCoreApplication.translate("Plotter", u"Level (% MVC)", None))
        self.lbl_live_level_duration_value.setText(QCoreApplication.translate("Plotter", u"Duration (s)", None))
        self.lbl_live_rest_value.setText(QCoreApplication.translate("Plotter", u"Rest (s)", None))
        self.lbl_live_repeat_value.setText(QCoreApplication.translate("Plotter", u"Repeats (#)", None))
        self.lbl_live_y_resolution.setText(QCoreApplication.translate("Plotter", u"Y Resolution (% MVC)", None))
        self.live_btn_preview.setText(QCoreApplication.translate("Plotter", u"Preview Profile", None))
        self.live_btn_start_measurement.setText(QCoreApplication.translate("Plotter", u"Start Online EMG", None))
        self.group_amplitude.setTitle(QCoreApplication.translate("Plotter", u"Amplitude", None))
        self.lbl_amplitude_UNI.setText(QCoreApplication.translate("Plotter", u"Amplitude UNI (\u03BCV)", None))
        self.label_logo.setText("")
        self.btn_hide_sidebar.setText(QCoreApplication.translate("Plotter", u"<<", None))
    # retranslateUi

