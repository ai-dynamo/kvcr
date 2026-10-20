# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic producer/consumer interleavings of the unit-test transport."""

from _kvcr_test_utils import FakeBytesControl


def test_fake_control_keeps_append_bound_before_argument_encoding():
    control = FakeBytesControl()

    def encode_while_receiver_runs() -> bytes:
        # The sender already holds incoming.append when its encoder yields.
        assert control.recv() == []
        return b"encoded-refusal"

    control.incoming.append(encode_while_receiver_runs())

    assert control.recv() == [b"encoded-refusal"]
    assert control.recv() == []


def test_fake_control_keeps_messages_appended_after_receive_snapshot():
    class AppendDuringCopy(list):
        appended = False

        def __getitem__(self, index):
            copied = super().__getitem__(index)
            if isinstance(index, slice) and not self.appended:
                self.appended = True
                self.append(b"arrived-during-receive")
            return copied

    control = FakeBytesControl()
    control.incoming = AppendDuringCopy([b"already-queued"])

    assert control.recv() == [b"already-queued"]
    assert control.recv() == [b"arrived-during-receive"]
    assert control.recv() == []
