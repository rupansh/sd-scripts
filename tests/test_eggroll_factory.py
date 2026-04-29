"""Verify the train_util.get_optimizer factory routes to EGGROLL.

This test imports the full training stack and so requires cv2/transformers/etc.
It is kept separate from test_eggroll.py so the pure optimizer tests can run
in environments that lack the training dependencies.
"""

from unittest.mock import patch

import pytest
import torch
from torch.nn import Parameter

cv2 = pytest.importorskip("cv2")  # skip if the imaging stack is missing

from library.eggroll_optimizer import EGGROLL  # noqa: E402
from library.train_util import get_optimizer  # noqa: E402
from train_network import setup_parser  # noqa: E402


def test_optimizer_factory_wires_eggroll():
    argv = [
        "",
        "--optimizer_type", "EGGROLL",
        "--optimizer_args",
        "population_size=8", "sigma=0.02", "seed=0",
    ]
    with patch("sys.argv", argv):
        parser = setup_parser()
        args = parser.parse_args()
        param = Parameter(torch.zeros(4, 4))
        name, _opt_args, optimizer = get_optimizer(args, [param])
        assert isinstance(optimizer, EGGROLL)
        assert getattr(optimizer, "is_eggroll_optimizer", False) is True
        assert "EGGROLL" in name.upper()
