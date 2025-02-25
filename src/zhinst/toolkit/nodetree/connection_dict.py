"""Implements Connection wrapper around a native python dictionary."""
import fnmatch
import json
import re
import typing as t
from collections import OrderedDict

from numpy import array

from zhinst.toolkit.nodetree.helper import NodeDoc


class ConnectionDict:
    """Connection wrapper around a dictionary.

    The ``NodeTree`` expects a connection that complies to the
    protocol :class:`nodetree.Connection`. In order to also support raw
    dictionaries this class wraps around a python dictionary and exposes the
    required protocol.

    Args:
        data: Dictionary raw path: value
        json_info: JSON information for each path (path: info)
    """

    def __init__(self, data: t.Dict[str, t.Any], json_info: NodeDoc):
        super().__init__()
        self._values = data
        self.json_info = json_info

    def listNodesJSON(self, path: str, *args, **kwargs) -> str:
        """Returns a list of nodes with description found at the specified path."""
        if path == "*":
            return json.dumps(self.json_info)
        json_info = {}
        for node, info in self.json_info.items():
            if fnmatch.fnmatchcase(node, path + "*"):
                json_info[node] = info
        return json.dumps(json_info)

    def get(self, path: str, *args, **kwargs) -> t.Any:
        """Mirrors the behavior of zhinst.core get command."""
        nodes_raw = fnmatch.filter(self._values.keys(), path)
        return_value = OrderedDict()
        for node in nodes_raw:
            return_value[node] = array([self._values[node]])
        return return_value

    def getInt(self, path: str) -> int:
        """Mirrors the behavior of zhinst.core getInt command."""
        try:
            return int(self._values[path])
        except TypeError:
            if self._values[path] is None:
                return 0
            raise

    def getDouble(self, path: str) -> float:
        """Mirrors the behavior of zhinst.core getDouble command."""
        return float(self._values[path])

    def getString(self, path: str) -> str:
        """Mirrors the behavior of zhinst.core getDouble command."""
        return str(self._values[path])

    def _parse_input_value(self, path: str, value: t.Any):
        if isinstance(value, str):
            option_map = {}
            for key, option in self.json_info[path].get("Options", {}).items():
                node_options = re.findall(r'"(.+?)"[,:]+', option)
                option_map.update({x: int(key) for x in node_options})
            return option_map.get(value, value)
        return value

    def set(
        self,
        path: t.Union[str, t.List[t.Tuple[str, t.Any]]],
        value: t.Any = None,
        **kwargs,
    ) -> None:
        """Mirrors the behavior of zhinst.core set command."""
        if isinstance(path, str):
            self._values[path] = self._parse_input_value(path, value)
        else:
            for node, node_value in path:
                self._values[node] = self._parse_input_value(node, node_value)

    def setVector(self, path: str, value: t.Any = None) -> None:
        """Mirrors the behavior of zhinst.core setVector command."""
        self.set(path, value)

    def subscribe(self, path: str) -> None:
        """Mirrors the behavior of zhinst.core subscribe command."""
        raise RuntimeError("Can not subscribe within the SHFQA_Sweeper")

    def unsubscribe(self, path: str) -> None:
        """Mirrors the behavior of zhinst.core unsubscribe command."""
        raise RuntimeError("Can not subscribe within the SHFQA_Sweeper")
