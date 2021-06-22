# Copyright (C) 2020 Zurich Instruments
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

from datetime import datetime
import re
import numpy as np
import deprecation

from ..interface import DeviceTypes
from .._version import version as __version__


class SequenceCommand(object):
    """A collection of sequence commands used for an AWG program."""

    @staticmethod
    @deprecation.deprecated(
        deprecated_in="0.2.0",
        current_version=__version__,
        details="Use the header_info function instead",
    )
    def header_comment(sequence_type="None"):
        now = datetime.now()
        now_string = now.strftime("%d/%m/%Y @%H:%M")
        return (
            f"// Zurich Instruments sequencer program\n"
            f"// sequence type:              {sequence_type}\n"
            f"// automatically generated:    {now_string}\n\n"
        )

    @staticmethod
    def header_info(sequence_type, trigger_mode, alignment):
        """Insert header information to the sequencer program.

        This function is used to display the sequence type, trigger mode and alignment
        information at the top of the sequencer program.

        Arguments:
            sequence_type (:class:`SequenceType` enum)
            trigger_mode (:class:`TriggerMode` enum)
            alignment (:class:`Alignment` enum)
        """
        now = datetime.now()
        now_string = now.strftime("%d/%m/%Y @%H:%M")
        return (
            f"// Zurich Instruments sequencer program\n"
            f"// sequence type:              {sequence_type.value}\n"
            f"// trigger mode:               {trigger_mode.value}\n"
            f"// alignment:                  {alignment.value}\n"
            f"// automatically generated:    {now_string}\n\n"
        )

    @staticmethod
    def replace_sequence_type(sequence, sequence_type):
        """Replace the sequence type in the header information of the sequencer
        program.

        The header information is initiated in the parent class with the
        default sequence type `None`. This default value should be replaced in
        the child class with the correct sequence type.

        Arguments:
            sequence (str): sequencer program that contains the initial header
            sequence_type (:class:`SequenceType` enum): correct sequence type

        """
        sequence_updated = re.sub(
            "// sequence type:              .*?\n",
            f"// sequence type:              {sequence_type.value}\n",
            sequence,
            flags=re.DOTALL,
        )
        return sequence_updated

    @staticmethod
    def repeat(i):
        if i == "inf":
            return f"while(true){{\n"
        if i < 0:
            raise ValueError("Invalid number of repetitions!")
        return f"repeat({int(i)}){{\n"

    @staticmethod
    def new_line():
        return "\n"

    @staticmethod
    def space():
        return " "

    @staticmethod
    def tab():
        return "\t"

    @staticmethod
    def comment_line():
        return "//\n"

    @staticmethod
    def inline_comment(comment):
        """Insert inline comment to the sequence.

        Arguments:
            comment (str): inline comment to be added
        """
        return f"// {comment}\n"

    @staticmethod
    def wait(i):
        """Inserts wait(...) command to the sequencer.

        Arguments:
            i (int): number of sequencer cycles to wait for

        """
        if i < 0:
            raise ValueError("Wait time cannot be negative!")
        if i == 0:
            return "//\n"
        else:
            return f"wait({int(i)});\n"

    @staticmethod
    def play_zero(i, target=DeviceTypes.HDAWG):
        """Inserts playZero(...) command to the sequencer.

        The granularity of the device will be automatically matched.

        Arguments:
            i (int): length in number of samples to play zero.
            target (str): type of the target device which
                determines the granularity to be matched.
                (default: DeviceTypes.HDAWG)

        """
        if i < 0:
            raise ValueError("Number of samples cannot be negative!")
        elif target in [DeviceTypes.HDAWG]:
            if i < 32:
                raise ValueError(
                    "Number of samples cannot be lower than 32 samples!")
            return f"playZero({int(round(i / 16) * 16)});\n"
        elif target in [DeviceTypes.UHFQA, DeviceTypes.UHFLI]:
            if i < 16:
                raise ValueError(
                    "Number of samples cannot be lower than 16 samples!")
            return f"playZero({int(round(i / 8) * 8)});\n"

    @staticmethod
    def wait_wave():
        return "waitWave();\n"

    @staticmethod
    @deprecation.deprecated(
        deprecated_in="0.2.0",
        current_version=__version__,
        details="Use the define_trigger and play_trigger functions instead",
    )
    def trigger(value, index=1):
        if value not in [0, 1]:
            raise ValueError("Invalid Trigger Value!")
        if index not in [1, 2]:
            raise ValueError("Invalid Trigger Index!")
        return f"setTrigger({value << (index - 1)});\n"

    @staticmethod
    def define_trigger(length=32):
        """Define a marker waveform to be used to send out trigger signals.

        The analog part of the waveform is zero.

        Arguments:
            length (int): length of marker waveform in number of samples. (default: 32)
        """
        if length < 32:
            raise ValueError("Trigger cannot be shorter than 32 samples!")
        if length % 16:
            raise ValueError("Trigger Length has to be multiple of 16!")
        return f"wave start_trigger = marker({length},1);\n"

    @staticmethod
    def play_trigger():
        """Play the marker waveform which is used as trigger."""
        return "playWave(1, start_trigger);\n"

    @staticmethod
    def count_waveform(i, n):
        return f"// waveform {i+1} / {n}\n"

    @staticmethod
    def assign_wave_index(i):
        """Assign an index to the labeled waveforms.

        Arguments:
            i (int): index to be assigned for the labeled waveforms.

        """
        if i < 0:
            raise ValueError("Waveform Index cannot be negative!")
        return f"assignWaveIndex(w{i + 1}_1, w{i + 1}_2, {i});\n"

    @staticmethod
    def play_wave():
        return "playWave(w_1, w_2);\n"

    @staticmethod
    def play_wave_scaled(amp1, amp2):
        if abs(amp1) > 1 or abs(amp2) > 1:
            raise ValueError("Amplitude cannot be larger than 1.0!")
        return f"playWave({amp1}*w_1, {amp2}*w_2);\n"

    @staticmethod
    def play_wave_indexed(i):
        if i < 0:
            raise ValueError("Invalid Waveform Index!")
        return f"playWave(w{i + 1}_1, w{i + 1}_2);\n"

    @staticmethod
    def play_wave_indexed_scaled(amp1, amp2, i):
        if i < 0:
            raise ValueError("Invalid Waveform Index!")
        if abs(amp1) > 1 or abs(amp2) > 1:
            raise ValueError("Amplitude cannot be larger than 1.0!")
        return f"playWave({amp1}*w{i+1}_1, {amp2}*w{i+2}_2);\n"

    @staticmethod
    def init_buffer_indexed(length, i, target=DeviceTypes.HDAWG):
        """Initialize placeholders (`placeholder(...)`) of specified length.

        The granularity of the device should be matched.

        Arguments:
            length (int): length of the placeholders in number of samples.
            i (int): index for the waveform label.
            target (str): type of the target device which
                determines the granularity to be matched.
                (default: DeviceTypes.HDAWG)

        """
        if i < 0:
            raise ValueError("Invalid Values for waveform buffer!")
        elif target in [DeviceTypes.HDAWG]:
            if length < 32:
                raise ValueError(
                    "Buffer Length cannot be lower than 32 samples!")
            elif length % 16:
                raise ValueError("Buffer Length has to be multiple of 16!")
        elif target in [DeviceTypes.UHFQA, DeviceTypes.UHFLI]:
            if length < 16:
                raise ValueError(
                    "Buffer Length cannot be lower than 16 samples!")
            elif length % 8:
                raise ValueError("Buffer Length has to be multiple of 8!")
        return (
            f"wave w{i + 1}_1 = placeholder({length});\n"
            f"wave w{i + 1}_2 = placeholder({length});\n"
        )

    @staticmethod
    def init_gauss(gauss_params):
        length, pos, width = gauss_params
        if length < 16:
            raise ValueError("Invalid Value for length!")
        if length % 16:
            raise ValueError("Length has to be multiple of 16!")
        if not (length > pos and length > width):
            raise ValueError(
                "Length has to be larger than position and width!")
        if not (width > 0):
            raise ValueError("Values cannot be negative!")
        return (
            f"wave w_1 = gauss({length}, {pos}, {width});\n"
            f"wave w_2 = gauss({length}, {pos}, {width});\n"
        )

    @staticmethod
    def init_gauss_scaled(amp, gauss_params):
        length, pos, width = gauss_params
        if abs(amp) > 1:
            raise ValueError("Amplitude cannot be larger than 1.0!")
        if length < 16:
            raise ValueError("Invalid Value for length!")
        if length % 16:
            raise ValueError("Length has to be multiple of 16!")
        if not (length > pos and length > width):
            raise ValueError(
                "Length has to be larger than position and width!")
        if not (width > 0):
            raise ValueError("Values cannot be negative!")
        return (
            f"wave w_1 = {amp} * gauss({length}, {pos}, {width});\n"
            f"wave w_2 = {amp} * drag({length}, {pos}, {width});\n"
        )

    @staticmethod
    def init_readout_pulse(length, amps, frequencies, phases, clk_rate=1.8e9):
        if len(frequencies) == 0:
            s = ""
            s += "wave w_1 = zeros(32);\n"
            s += "wave w_2 = zeros(32);\n"
            s += "\n"
            return s
        assert len(amps) == len(frequencies)
        assert len(phases) == len(frequencies)
        assert abs(max(amps)) <= 1.0
        n_periods = [length * f / clk_rate for f in frequencies]
        n = len(n_periods)
        s = str()
        for i in range(n):
            s += f"wave w{i+1}_I = 1/{n} * sine({int(length)}, {amps[i]}, 0, {n_periods[i]});\n"
            s += f"wave w{i+1}_Q = 1/{n} * cosine({int(length)}, {amps[i]}, {np.deg2rad(phases[i])}, {n_periods[i]});\n"
        s += "\n"
        if n > 1:
            s += (
                f"wave w_1 = add(" +
                ", ".join([f"w{i+1}_I" for i in range(n)]) + ");\n"
            )
            s += (
                f"wave w_2 = add(" +
                ", ".join([f"w{i+1}_Q" for i in range(n)]) + ");\n"
            )
        else:
            s += "wave w_1 = w1_I;\n"
            s += "wave w_2 = w1_Q;\n"
        s += "\n"
        return s

    @staticmethod
    def close_bracket():
        return "}"

    @staticmethod
    def wait_dig_trigger(index=1, target=DeviceTypes.HDAWG):
        """Insert waitDigTrigger(...) command to the sequencer.

        The arguments of waitDigTrigger(...) function are different
        for HDAWG and UHFQA/UHFLI.

        Arguments:
            index (int): index of the digital trigger input;
                can be either 1 or 2. (default: 1)
            target (str): type of the target device which
                determines the arguments to pass to the
                function. (default: DeviceTypes.HDAWG)
        """

        if index not in [1, 2]:
            raise ValueError("Invalid Trigger Index!")
        if target in [DeviceTypes.HDAWG, DeviceTypes.SHFQA]:
            return f"waitDigTrigger({index});\n"
        elif target in [DeviceTypes.UHFQA, DeviceTypes.UHFLI]:
            return f"waitDigTrigger({index}, 1);\n"

    @staticmethod
    def wait_zsync_trigger():
        """Insert waitZSyncTrigger(...) command to the sequencer."""
        return "waitZSyncTrigger();\n"

    @staticmethod
    def readout_trigger():
        """Start the Quantum Analyzer Result and Input units.

        Reads only qubit result 1 and 2 among ten possible results.

        """
        return "startQA(QA_INT_0 | QA_INT_1, true);\n"

    @staticmethod
    def init_ones(amp, length):
        return (
            f"wave w_1 = {amp} * ones({length});\n"
            f"wave w_2 = {amp} * ones({length});\n\n"
        )

    @staticmethod
    def reset_osc_phase():
        return f"resetOscPhase();\n"
