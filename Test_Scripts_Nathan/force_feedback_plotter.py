'''
(c) 2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file Force_feedback_plotter.py 
 * @brief 
 * ForceFeedbackPlotter object.
 */


'''
import numpy as np

from PySide2 import QtWidgets

from Test_Scripts_Nathan.experiment_plotter import Experiment_Plotter
from Test_Scripts_Nathan.force_feedback_chart import ForceFeedbackChart
from TMSiFrontend.components.channel_component import ChannelComponent
from TMSiSDK.device.tmsi_device_enums import ChannelType


class ForceFeedbackPlotter(Experiment_Plotter):
    """ForceFeedbackPlotter object
    """
    def __init__(self, meas_type , get_data_callback = None, name = "ForceForceFeedbackPlotter"):
        """Initialize the SignalPlotter object

        :param get_data_callback: function to be called when the interface is updated (used in case of offline application), defaults to None
        :type get_data_callback: function, optional
        :param name: name of the signal plotter, defaults to "Signal Plotter"
        :type name: str, optional
        """
        super().__init__(meas_type= meas_type, name = name)
        self.chart = ForceFeedbackChart(self.chart)
        self._get_data_callback = get_data_callback
        self._update_enabled_disabled = False
        self._update_scales_disabled = False
        self._compute_autoscale = False
        self.window_size = 10
        self.time_elapsed = 0
        self.chart.set_time_range(self.window_size)
        self._connect_widgets_to_functions()
        self.scale = 10
        self.update_timer_true = False
        self.round = 0
        self.size_buffer = 40000
        
    def autoscale(self):
        """Autoscale amplitudes of the channels
        """
        self._compute_autoscale = True
        self._update_data()
        
    def enable_all_channels(self, enabled = True):
        """Enable all channels to be seen

        :param enabled: True if to enable, False otherwise, defaults to True
        :type enabled: bool, optional
        """
        self._update_enabled_disabled = True
        
        for scroll_child in self.scrollAreaWidgetContents.children():
            if isinstance(scroll_child, QtWidgets.QFrame):
                for i in scroll_child.children():
                    if isinstance(i, QtWidgets.QCheckBox):
                        i.setChecked(enabled)
        self._update_enabled_disabled = False
        self.update_enabled_channels()

    def enable_channels(self, channels = ['AUX 1-2', 'Force Profile', 'Motor Unit 1', 'Motor Unit 2']):
        """Enable all channels to be seen

        :param enabled: True if to enable, False otherwise, defaults to True
        :type enabled: bool, optional
        """
        self._update_enabled_disabled = True
        j = 0 
        for scroll_child in self.scrollAreaWidgetContents.children():
            if isinstance(scroll_child, QtWidgets.QFrame):
                for i in scroll_child.children():
                    if isinstance(i, QtWidgets.QCheckBox):
                        component_instance = getattr(self, f"component_channel_{j}")
                        if isinstance(component_instance, ChannelComponent):
                            if component_instance.get_name() in channels:
                                i.setChecked(True)
                            else:
                                i.setChecked(False)
                        j += 1
        self._update_enabled_disabled = False
        self.update_enabled_channels()

    def initialize_channels_components(self, channels):
        """Initialize channels components

        :param channels: list of channels to be plotted and controlled. Can be TMSiChannel or just the channel name as string
        :type channels: list[TMSiChannel] or list[str]
        """
        # remove any possible channel component
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        for cc in channel_components:
            getattr(self,cc).delete()
            self.__dict__.pop(cc, None)
        # initialize new components
        self._channel_names = {}
        self._channel_units = {}
        self._channel_types = {}
        for i in range(len(channels)):
            if isinstance(channels[i], str):
                # If channel is not an device channel, the channel type, unit and name will be set according to the following code
                # Channel type and unit is necessary for the xdf file writer. 
                channel_name = channels[i]
                channel_unit = "a.u."
                channel_type = ChannelType.unknown 
                enable = True
            else:
                channel_name = channels[i].get_channel_name()
                channel_unit = channels[i].get_channel_unit_name()
                channel_type = channels[i].get_channel_type()
                if channels[i].get_channel_type().value <= 4:
                    enable = True
                else:
                    enable = True  #aangepast om ze in de data te krijgen. 
            if hasattr(self,"component_channel_{}".format(i)):
                getattr(self,"component_channel_{}".format(i)).set_text("{}".format(channel_name))
                continue
            component = ChannelComponent(
                index = i,
                layout = self.layout_channels,
                widget = self.scrollAreaWidgetContents,
                connect_chb = lambda: self.update_enabled_channels(),
                checked_state = enable,
                connect_offset = lambda: self.update_offsets(),
                connect_scale = lambda: self.update_scales(),
                connect_combo = lambda: self.update_enabled_channels(),
                name = channel_name
            )
            setattr(self,"component_channel_{}".format(i), component)
            self._channel_names[i] = channel_name
            self._channel_types[i] = channel_type
            self._channel_units[i] = channel_unit
        # set compute autoscale to avoid the tentative to plot without offset available
        self._compute_autoscale = True
        self.update_enabled_channels()
        self.update_scales()
        self._compute_autoscale = False
        self.update_offsets()
        self.chart.update_y_ticks(
            list_names = self._filter_lists_to_plot(self._channel_names),
            list_offsets = self._filter_lists_to_plot(self._offsets),
            list_scales = self._filter_lists_to_plot(self._scales),
            list_units = self._filter_lists_to_plot(self._channel_units))
        list_names = self._filter_lists_to_plot(self._channel_names)
        list_units = self._filter_lists_to_plot(self._channel_units)
        list_types = self._filter_lists_to_plot(self._channel_types)
        return list_names, list_units, list_types


    def manual_scale(self):
        """Manual set of the scale value"""
        new_val = self.spin_amplitude.value()
        self._update_scales_disabled = True
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        for channel_component in channel_components:
            cmp = getattr(self,channel_component)
            cmp.set_scale(new_val)
        self._update_scales_disabled = False
        self.update_scales()
    
    def update_chart(self, data_to_plot, time_span = None):
        """Update chart

        :param data_to_plot: data to be plotted
        :type data_to_plot: list[list]
        :param time_span: time span for the channels (None to be automatically plotted in the range), defaults to None
        :type time_span: list, optional
        """
        if not self.is_chart_update_enabled:
            return
        #if self._compute_autoscale: 
        # The y-range gets changed each iteration, so autoscale does not work any more.
        self._update_scales_disabled = True
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        scales = []
        offsets = []
        for n_channel in range(len(data_to_plot)):
            # offset is the value of the last channel (so in most cases the value of the force profile).
            # This leads the plot to follow the force profile value over time. 
            offset = data_to_plot[-1,int(self.size_buffer/2)]
            if self.scale <= 1e-3:
                self.scale = 1
            # For every channel a scale of self.scale is used. This self.scale can be changed in the forcefeedbackplotterhelper>
            # In MVC, the self.scale is set to 100. In a training, the scale can be changed manually. 
            scales.append(self.scale)
            offsets.append(offset)
        for n_channel in range(len(channel_components)):
            cmp = getattr(self, channel_components[n_channel])
            cmp.set_scale(scales[cmp.get_index()])
            cmp.set_offset(offsets[cmp.get_index()])
        self._update_scales_disabled = False
        self.update_offsets()
        self.update_scales()
        self.chart.update_y_ticks(
            list_names = self._filter_lists_to_plot(self._channel_names),
            list_offsets = self._filter_lists_to_plot(self._offsets),
            list_scales = self._filter_lists_to_plot(self._scales),
            list_units = self._filter_lists_to_plot(self._channel_units))
        self._compute_autoscale = False
        for i in range(len(data_to_plot)):
            data_to_plot[i] = - (data_to_plot[i] - self._offsets[i]) / self._scales[i]
        self.chart.update_chart(self._filter_data_to_plot(data_to_plot), time_span)
    
        
    def update_colors(self):
        """Update color of the chart based on the channel component
        """
        colors = {}
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        for channel_component in channel_components:
            cmp = getattr(self,channel_component)
            colors[cmp.get_index()] = cmp.get_color()
        colors = [colors[i] for i in range(len(colors)) if i in self._enabled_channels]
        self.chart.setup_signals(n_signals=len(colors), colors=colors)

    def update_enabled_channels(self):
        """Update the enabled channels based on channel component
        """
        if self._update_enabled_disabled:
            return
        self._enabled_channels = []
        for scroll_child in self.scrollAreaWidgetContents.children():
            if isinstance(scroll_child, QtWidgets.QFrame):
                for i in scroll_child.children():
                    if isinstance(i, QtWidgets.QCheckBox):
                        if i.isChecked():
                            self._enabled_channels.append(int(i.objectName().split("_")[-1]))
        self.update_colors()
        if not self._compute_autoscale:
            self._update_data()
            self.chart.update_y_ticks(
                list_names = self._filter_lists_to_plot(self._channel_names),
                list_offsets = self._filter_lists_to_plot(self._offsets),
                list_scales = self._filter_lists_to_plot(self._scales),
                list_units = self._filter_lists_to_plot(self._channel_units))

    def update_offsets(self):
        """Update the offset based on channel component
        """
        if self._update_scales_disabled:
            return
        self._offsets = {}
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        for channel_component in channel_components:
            cmp = getattr(self,channel_component)
            self._offsets[cmp.get_index()] = cmp.get_offset()
        if not self._compute_autoscale:
            self._update_data()
            self.chart.update_y_ticks(
                list_names = self._filter_lists_to_plot(self._channel_names),
                list_offsets = self._filter_lists_to_plot(self._offsets),
                list_scales = self._filter_lists_to_plot(self._scales),
                list_units = self._filter_lists_to_plot(self._channel_units))
        
    def update_scales(self):
        """Update the scale based on channel component
        """
        if self._update_scales_disabled:
            return
        self._scales = {}
        channel_components = [i for i in dir(self) if i.startswith("component_channel_")]
        for channel_component in channel_components:
            cmp = getattr(self,channel_component)
            self._scales[cmp.get_index()] = cmp.get_scale()
        if not self._compute_autoscale:
            self._update_data()
            self.chart.update_y_ticks(
                list_names = self._filter_lists_to_plot(self._channel_names),
                list_offsets = self._filter_lists_to_plot(self._offsets),
                list_scales = self._filter_lists_to_plot(self._scales),
            list_units = self._filter_lists_to_plot(self._channel_units))

    def update_time_ticks(self, start_time, end_time):
        """Update time ticks

        :param start_time: initial time value
        :type start_time: float
        :param end_time: final time tick
        :type end_time: float
        """
        self.chart.update_time_ticks(start_time=start_time, end_time=end_time)

    def _connect_widgets_to_functions(self):
        self.btn_enable_all_channels.clicked.connect(lambda: self.enable_all_channels(True))
        self.btn_disable_all_channels.clicked.connect(lambda: self.enable_all_channels(False))
        self.btn_autoscale.clicked.connect(self.autoscale)
        self.spin_amplitude.valueChanged.connect(self.manual_scale)

    def _filter_data_to_plot(self, data_to_plot):
        data_to_plot = [data_to_plot[row] for row in range(len(data_to_plot)) if row in self._enabled_channels]
        return data_to_plot
    
    def _filter_lists_to_plot(self, list_to_filter):
        return [list_to_filter[i] for i in range(len(list_to_filter)) if i in self._enabled_channels]
    
    def _update_data(self):
        if self._get_data_callback is not None:
            self._get_data_callback()

    def update_timer(self):
        # Function to update the timer. It changes the value based on the pointer of the datastream. 
        # It also changes the layout when MVC must be done.
        if self.update_timer_true and self.round < self.mvc_repeats.value(): 
            self.round_time = self.mvc_duration.value() + self.mvc_rest.value()
            if self.time_elapsed < self.start_rest: 
                self.displayed_time = self.start_rest - self.time_elapsed
                self.lbl_timer_value.setText(f"{round(self.displayed_time,2)}")
                self.lbl_timer_value.setStyleSheet("color: black; font-size: 32px")
            elif self.time_elapsed < self.start_rest + self.mvc_duration.value() + self.round*self.round_time: 
                self.displayed_time = self.start_rest + self.mvc_duration.value() + self.round*self.round_time - self.time_elapsed
                self.lbl_timer_value.setText(f"{round(self.displayed_time,2)}")
                self.lbl_timer_value.setStyleSheet("color: red; font-size: 32px") # Make the text red
            elif self.time_elapsed < self.start_rest + self.mvc_duration.value() + self.mvc_rest.value() + self.round*self.round_time :
                self.displayed_time = self.start_rest + self.mvc_duration.value() + self.mvc_rest.value() + self.round*self.round_time - self.time_elapsed
                self.lbl_timer_value.setText(f"{round(self.displayed_time,2)}")
                self.lbl_timer_value.setStyleSheet("color: black; font-size: 32px")  # Make text black again
            else:
                self.round += 1 
                if self.round < self.mvc_repeats.value():
                    self.displayed_time = self.start_rest + self.mvc_duration.value() + self.round*self.round_time - self.time_elapsed
                    self.lbl_timer_value.setText(f"{round(self.displayed_time,2)}")
                    self.lbl_timer_value.setStyleSheet("color: red; font-size: 32px")
                else:
                    self.displayed_time = ''
                    self.lbl_timer_value.setText(f"MVC Done")
                    self.lbl_timer_value.setStyleSheet("color: red; font-size: 32px")

    def update_offset_label(self):
        # Decrement the counter value
        self.calibration_seconds -= 1

        if self.calibration_seconds > 0:  # Update the label text
            self.lbl_calibration_timer.setText(f"{round(self.calibration_seconds)}")
            self.lbl_calibration_timer.setStyleSheet("color: black; font-size: 32px")
        elif self.calibration_seconds > -5:
            self.lbl_calibration_timer.setText(f"PUSH NOW")
            self.lbl_calibration_timer.setStyleSheet("color: red; font-size: 32px")
        elif self.calibration_seconds < -5:
            self.lbl_calibration_timer.setText(f"Calibration Done")
            self.lbl_calibration_timer.setStyleSheet("color: black; font-size: 32px")


        
        