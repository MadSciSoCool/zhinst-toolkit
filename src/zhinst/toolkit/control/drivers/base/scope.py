from logging import warn
import time
import numpy as np
from typing import List, Dict

from .base import ToolkitError, BaseInstrument
from ...node_tree import Parameter

MAPPINGS = {
    "averager_resamplingmode": {
        0: "linear",
        1: "pchip"
    },
    "mode": {
        0: "passthrough",
        1: "exp_moving_average",
        # 2: "reserved",
        3: "fft"
    },
    "fft_window": {
        0: "rectangular",
        1: "hann",
        2: "hamming",
        3: "blackman",
        16: "exponential",
        17: "cosine",
        17: "sine",
        18: "cosine_squared",
    },
    "save_fileformat": {
        0: "matlab",
        1: "csv",
        2: "zview",
        3: "sxm",
        4: "hdf5",
    }
}


class ScopeModule:
    """Implements a :class:`Scope Module` for UHFQA instruments.

    In a typical measurement using the scope Module one would first configure its
    settings.

        >>> # configure a measurement (in the scope module)
        >>> qa.scope.mode("passthrough")
        >>> # configure a measurement (in the qa nodetree)
        >>> qa.nodetree.scope.length(2 ** 12)
        >>> qa.nodetree.scope.time(2)

    The measurement is started ...
        >>> # start the measurement
        >>> qa.scope.measure()
        subscribed to: /dev2571/scopes/0/wave
        Progress: 0.0%
        Finished

    ... and afterwards the results can be found in the `results` attribute of
    the :class:`ScopeModule`. The values in the dictionary are of type
    :class:`ScopeResults`.
        >>> # retrieve the measurement results
        >>> results = qa.scope.results
        >>> results
        <zhinst_test.toolkit.control.drivers.base.scope.ScopeResult object at 0x000001A34E9DCE80>
            path:        /dev2571/scopes/0/wave
            value:       (1, 4096)
            time:        (4096,)


    Attributes:
        results (dict): A dictionary with signal strings as keys and
            :class:`zhinst.toolkit.control.drivers.base.daq.DAQResult` objects
            as values that hold all the data of the measurement result.

    """

    def __init__(self, parent: BaseInstrument) -> None:
        self._parent = parent
        self._module = None
        self._results = None
        self._path = f'/{self._parent.serial}/scopes/0/wave'

    def _setup(self) -> None:
        self._module = self._parent._controller._connection._scope_module
        nodetree = self._module.get_nodetree("*")
        for k, v in nodetree.items():
            name = k[1:].replace("/", "_")
            mapping = MAPPINGS[name] if name in MAPPINGS.keys() else None
            setattr(self, name, Parameter(
                self, v, device=self, mapping=mapping))
        self._init_settings()

    def _set(self, *args):
        if self._module is None:
            raise ToolkitError(
                "This DAQ is not connected to a scopeModule!")
        return self._module.set(*args)

    def _get(self, *args, valueonly: bool = True):
        if self._module is None:
            raise ToolkitError(
                "This DAQ is not connected to a scopeModule!")
        data = self._module.get(*args)
        return list(data.values())[0][0] if valueonly else data

    def measure(self, verbose: bool = True, timeout: float = 20, blocking: bool = True) -> None:
        """Performs the measurement.

        Starts a measurement and stores the result in `scope.results`. This
        method subscribe to the scole signal of the given device, then 
        starts the measurement, waits until the measurement in finished and 
        eventually reads the result.

        Keyword Arguments:
            verbose (bool): A flag to enable or disable console output during
                the measurement. (default: True)
            timeout (int): The measurement will be stopped after the timeout.
                The valiue is given in seconds. (default: 20)

        """
        self._set("clearhistory", 1)
        self._set("averager/restart", 1)
        self._parent._set("scopes/0/enable", 0)
        # a hardware bug that makes the scope stuck if set /scopes/0/enable to be 1 twice
        self._parent._set("scopes/0/enable", 1)
        self._module.subscribe(self._path)
        if verbose:
            print(f"subscribed to: {self._path}")
        self._module.execute()
        self._tik = time.time()
        self._timeout = timeout
        if blocking:
            while self.progress != 1:
                if verbose:
                    print(
                        f"Progress: {(self._module.progress()[0] * 100):.1f}%")
                time.sleep(0.5)
                # tok = time.time()
                # if tok - self._tik > timeout:
                #     raise TimeoutError()
            if verbose:
                print("Finished")
            self.finish()
        else:
            if verbose:
                print("""The scope is now running in non-blocking mode: 
                      use scope.progress to check the progress (0 to 1)
                      and execute scope.finish() to save the result""")

    def finish(self):
        if self._module.progress() == 1:
            result = self._module.read()  # flat=True
            self._module.finish()
            self._module.unsubscribe("*")
            trig_reference = self._parent._get("scopes/0/trigreference")
            trig_delay = self._parent._get("scopes/0/trigdelay")
            self._results = ScopeResult(path=self._path,
                                        result_dict=result[self._parent.serial]['scopes']['0']['wave'][0][0],
                                        clk_rate=self._clk_rate,
                                        acquisition_mode=self._get("mode"),
                                        trig_reference=trig_reference,
                                        trig_delay=trig_delay)
        else:
            warn("""The scope has not finished reading, the result is not saved""")

    @property
    def progress(self):
        tok = time.time()
        if tok - self._tik > self._timeout:
            raise TimeoutError()
        return self._module.progress()

    @property
    def _clk_rate(self):
        node = "scopes/0/time"
        base = self._parent._get(node)
        return 1.8e9 / (2 ** base)

    @property
    def results(self):
        return self._results

    def _init_settings(self):
        pass


class ScopeResult:
    """A wrapper class around the result of a DAQ Module measurement.

    The Data Acquisition Result class holds all measurement information returned
    from the API. The attribute `value` is a two-dimensional numpy array with
    the measured data along the measured grid. Depending on whether the time
    trace or the FFT of a signal was acquired, either the `time` of `frequency`
    attribute holds a 1D numpy array with the correct axis values calculated
    from the measurement grid.

        >>> qa.scope.measure()
        ...
        >>> result = qa.scope.results
        >>> result
        <zhinst_test.toolkit.control.drivers.base.scope.ScopeResult object at 0x000001A34E017D00>
            path:        /dev2571/scopes/0/wave
            value:       (1, 4096)
            time:        (4096,)
        >>> result.header
           {'systemtime': array([1623433101011891], dtype=uint64),
            'createdtimestamp': array([1097810349264496], dtype=uint64),
            'changedtimestamp': array([1097810349264496], dtype=uint64),
            'flags': array([57], dtype=uint32),
            ...

    Attributes:
        value (array): A 2D numpy array with the measurement result.
        shape (tuple): A tuple with the shape of the acquired data which
            corresponds to the according grid settings.
        time (array): A 1D numpy array containing the time axis of the
            measurement in seconds. Calculated from the returned timestamps
            using the DAC clock rate. If the result is a Fourier transform this
            value is `None`.
        frequency (array): A 1D numpy array with the frequency values for FFT
            measurements in Hertz.
        header (dict): A dictionary containing all information about the
            measurement settings.

    """

    def __init__(self, path: str, result_dict: Dict, clk_rate: float = 1.8e9, acquisition_mode: int = 0, trig_reference: float = 0.5, trig_delay: float = 0) -> None:
        self._path = path
        self._clk_rate = clk_rate
        self._result_dict = result_dict
        self._header = self._result_dict.get("header", {})
        self._value = self._result_dict.get("wave")
        self._time = None
        self._frequencies = None
        if acquisition_mode == 0:
            self._value = self._value / 2 ** 15
        self._is_fft = acquisition_mode == 3
        if not self._is_fft:
            self._time = self._calculate_time(trig_reference, trig_delay)
        else:
            self._frequency = self._calculate_freqs()

    @property
    def value(self):
        return self._value

    @property
    def header(self):
        return self._header

    @property
    def time(self):
        return self._time

    @property
    def frequency(self):
        return self._frequency

    @property
    def shape(self):
        return self._value.shape

    def _calculate_time(self, trig_reference, trig_delay):
        raw_time = np.arange(self._result_dict.get(
            "totalsamples")) / self._clk_rate
        amp = raw_time[-1] / 2
        offset = 2 * trig_reference * amp - trig_delay
        return raw_time - offset

    def _calculate_freqs(self):
        bin_count = self._result_dict.get("totalsamples")
        max_freq = self._clk_rate / 2
        frequencies = max_freq * np.linspace(0, 1, bin_count)
        return frequencies

    def __repr__(self):
        s = super().__repr__()
        s += "\n\n"
        s += f"path:        {self._path}\n"
        s += f"value:       {self._value.shape}\n"
        if self._is_fft:
            s += f"frequency:   {self._frequency.shape}\n"
        else:
            s += f"time:        {self._time.shape}\n"
        return s
