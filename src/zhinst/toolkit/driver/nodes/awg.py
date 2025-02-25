"""zhinst-toolkit AWG node adaptions."""
import json
import logging
import typing as t

from zhinst.core import compile_seqc
from zhinst.toolkit.driver.nodes.command_table_node import CommandTableNode
from zhinst.toolkit.nodetree import Node, NodeTree
from zhinst.toolkit.nodetree.helper import (
    lazy_property,
    create_or_append_set_transaction,
)
from zhinst.toolkit.waveform import Waveforms
from zhinst.toolkit.sequence import Sequence

logger = logging.getLogger(__name__)


class AWG(Node):
    """AWG node.

    This class implements the basic functionality for the device specific
    arbitrary waveform generator.
    Besides the upload/compilation of sequences it offers the upload of
    waveforms and command tables.

    Args:
        root: Root of the nodetree
        tree: Tree (node path as tuple) of the current node
        session: Underlying session.
        serial: Serial of the device.
        index: Index of the corresponding awg channel
        device_type: Device type
    """

    def __init__(
        self,
        root: NodeTree,
        tree: tuple,
        serial: str,
        index: int,
        device_type: str,
        device_options: str,
    ):
        Node.__init__(self, root, tree)
        self._daq_server = root.connection
        self._serial = serial
        self._index = index
        self._device_type = device_type
        self._device_options = device_options

    def enable_sequencer(self, *, single: bool) -> None:
        """Starts the sequencer of a specific channel.

        Waits until the sequencer is enabled.

        Args:
            single: Flag if the sequencer should be disabled after finishing
            execution.
        """
        self.single(single)
        self.enable(1, deep=True)
        self.enable.wait_for_state_change(1)

    def wait_done(self, *, timeout: float = 10, sleep_time: float = 0.005) -> None:
        """Wait until the AWG is finished.

        Args:
            timeout: The maximum waiting time in seconds for the generator
                (default: 10).
            sleep_time: Time in seconds to wait between requesting generator
                state

        Raises:
            RuntimeError: If continuous mode is enabled
            TimeoutError: If the sequencer program did not finish within
                the specified timeout time
        """
        if not self.single():
            raise RuntimeError(
                f"{repr(self)}: The generator is running in continuous mode, "
                "it will never be finished."
            )
        try:
            self.enable.wait_for_state_change(0, timeout=timeout, sleep_time=sleep_time)
        except TimeoutError as error:
            raise TimeoutError(
                f"{repr(self)}: The execution of the sequencer program did not finish "
                f"within the specified timeout ({timeout}s)."
            ) from error

    def compile_sequencer_program(
        self,
        sequencer_program: t.Union[str, Sequence],
        **kwargs: t.Union[str, int],
    ) -> t.Tuple[bytes, t.Dict[str, t.Any]]:
        """Compiles a sequencer program for the specific device.

        Args:
            sequencer_program: The sequencer program to compile.

        Keyword Args:
            samplerate: Target sample rate of the sequencer. Only allowed/
                necessary for HDAWG devices. Must correspond to the samplerate
                used by the device (device.system.clocks.sampleclock.freq()).
                If not specified the function will get the value itself from
                the device. It is recommended passing the samplerate if more
                than one sequencer code is uploaded in a row to speed up the
                execution time.
            wavepath: path to directory with waveforms. Defaults to path used
                by LabOne UI or AWG Module.
            waveforms: waveform CSV files separated by ';'
            output: name of embedded ELF filename.

        Returns:
            elf: Binary ELF data for sequencer.
            extra: Extra dictionary with compiler output.

        Examples:
            >>> elf, compile_info = device.awgs[0].compile_sequencer_program(seqc)
            >>> device.awgs[0].elf.data(elf)
            >>> device.awgs[0].ready.wait_for_state_change(1)
            >>> device.awgs[0].enable(True)

        Raises:
            RuntimeError: `sequencer_program` is empty.
            RuntimeError: If the compilation failed.

        .. versionadded:: 0.4.0
        """
        if "SHFQC" in self._device_type:
            kwargs["sequencer"] = "sg" if "sgchannels" in self._tree else "qa"
        elif "HDAWG" in self._device_type and "samplerate" not in kwargs:
            kwargs["samplerate"] = self.root.system.clocks.sampleclock.freq()

        return compile_seqc(
            str(sequencer_program),
            self._device_type,
            self._device_options,
            **kwargs,
        )

    def load_sequencer_program(
        self,
        sequencer_program: t.Union[str, Sequence],
        **kwargs: t.Union[str, int],
    ) -> t.Dict[str, t.Any]:
        """Compiles the given sequencer program on the AWG Core.

        Warning:
            After uploading the sequencer program one needs to wait before for
            the awg core to become ready before it can be enabled.
            The awg core indicates the ready state through its `ready` node.
            (device.awgs[0].ready() == True)

        Args:
            sequencer_program: Sequencer program to be uploaded.

        Keyword Args:
            samplerate: Target sample rate of the sequencer. Only allowed/
                necessary for HDAWG devices. Must correspond to the samplerate
                used by the device (device.system.clocks.sampleclock.freq()).
                If not specified the function will get the value itself from
                the device. It is recommended passing the samplerate if more
                than one sequencer code is uploaded in a row to speed up the
                execution time.
            wavepath: path to directory with waveforms. Defaults to path used
                by LabOne UI or AWG Module.
            waveforms: waveform CSV files separated by ';'
            output: name of embedded ELF filename.

        Examples:
            >>> compile_info = device.awgs[0].load_sequencer_program(seqc)
            >>> device.awgs[0].ready.wait_for_state_change(1)
            >>> device.awgs[0].enable(True)

        Raises:
            RuntimeError: `sequencer_program` is empty.
            RuntimeError: If the upload or compilation failed.

        .. versionadded:: 0.3.4

            `sequencer_program` does not accept empty strings

        .. versionadded:: 0.4.0

            Use offline compiler instead of AWG module to compile the sequencer
            program. This speeds of the compilation and also enables parallel
            compilation/upload.
        """
        elf, compiler_info = self.compile_sequencer_program(sequencer_program, **kwargs)
        self.elf.data(elf)
        return compiler_info

    def write_to_waveform_memory(
        self, waveforms: Waveforms, indexes: list = None, validate: bool = True
    ) -> None:
        """Writes waveforms to the waveform memory.

        The waveforms must already be assigned in the sequencer program.

        Args:
            waveforms: Waveforms that should be uploaded.
            indexes: Specify a list of indexes that should be uploaded. If
                nothing is specified all available indexes in waveforms will
                be uploaded. (default = None)
            validate: Enable sanity check preformed by toolkit, based on the
                waveform descriptors on the device. Can be disabled for e.g.
                speed optimizations. Does not affect the checks happen in LabOne
                and or the firmware. (default = True)

        Raises:
            IndexError: The index of a waveform exceeds the one on the device
                and `validate` is True.
            RuntimeError: One of the waveforms index points to a
                filler(placeholder) and `validate` is True.
        """
        waveform_info = None
        num_waveforms = None
        if validate:
            waveform_info = json.loads(self.waveform.descriptors()).get("waveforms", [])
            num_waveforms = len(waveform_info)
        with create_or_append_set_transaction(self._root):
            for waveform_index in waveforms.keys():
                if indexes and waveform_index not in indexes:
                    continue
                if num_waveforms is not None and waveform_index >= num_waveforms:
                    raise IndexError(
                        f"There are {num_waveforms} waveforms defined on the device "
                        "but the passed waveforms specified one with index "
                        f"{waveform_index}."
                    )
                if (
                    waveform_info
                    and "__filler" in waveform_info[waveform_index]["name"]
                ):
                    raise RuntimeError(
                        f"The waveform at index {waveform_index} is only "
                        "a filler and can not be overwritten"
                    )
                self.root.transaction.add(
                    self.waveform.waves[waveform_index],
                    waveforms.get_raw_vector(
                        waveform_index,
                        target_length=int(waveform_info[waveform_index]["length"])
                        if waveform_info
                        else None,
                    ),
                )

    def read_from_waveform_memory(self, indexes: t.List[int] = None) -> Waveforms:
        """Read waveforms to the waveform memory.

        Args:
            indexes: List of waveform indexes to read from the device. If not
                specified all assigned waveforms will be downloaded.

        Returns:
            Waveform object with the downloaded waveforms.
        """
        nodes = [self.waveform.descriptors.node_info.path]
        if indexes is not None:
            for index in indexes:
                nodes.append(self.waveform.node_info.path + f"/waves/{index}")
        else:
            nodes.append(self.waveform.waves["*"].node_info.path)
        nodes_str = ",".join(nodes)
        waveforms_raw = self._daq_server.get(nodes_str, settingsonly=False, flat=True)
        waveform_info = json.loads(
            waveforms_raw.pop(self.waveform.descriptors.node_info.path)[0]["vector"]
        ).get("waveforms", [])
        waveforms = Waveforms()
        for node, waveform in waveforms_raw.items():
            slot = int(node[-1])
            if "__filler" not in waveform_info[slot]["name"]:
                waveforms.assign_native_awg_waveform(
                    slot,
                    waveform[0]["vector"],
                    channels=int(waveform_info[slot].get("channels", 1)),
                    markers_present=bool(
                        int(waveform_info[slot].get("marker_bits")[0])
                    ),
                )
        return waveforms

    @lazy_property
    def commandtable(self) -> t.Optional[CommandTableNode]:
        """Command table module."""
        if self["commandtable"].is_valid():
            return CommandTableNode(
                self._root, self._tree + ("commandtable",), self._device_type
            )
        return None
