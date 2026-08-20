# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduction test for ring buffer deadlock caused by lost ZMQ notifications.

When the reader is in the SpinCondition hot-path (within busy_loop_s of the
last read), it only calls sched_yield() and never polls the ZMQ notification
socket.  If the writer calls notify() during this window the zmq.CONFLATE
setting can discard the notification.  The reader then enters the cold-path
and blocks on the poller waiting for a notification that never comes; on the
writer side acquire_write() spins waiting for the reader's ACK.

The fix adds a non-blocking ZMQ poll at the start of every hot-path wait()
call so that notifications are never lost.
"""

import threading
import time
from unittest import mock

import zmq

from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    SpinCondition,
)


def test_spincondition_hot_path_notify_not_lost():
    """Verify that notify() sent while the reader is in hot-path is still
    visible via a non-blocking poll.  Without the fix the notify is lost
    because wait() only calls sched_yield() during the busy-loop window."""
    ctx = zmq.Context()
    addr = "inproc://test_spincondition_hot_path"

    # Writer binds first (PUB), then reader connects (SUB)
    writer_cond = SpinCondition(
        is_reader=False, context=ctx, notify_address=addr
    )
    reader_cond = SpinCondition(
        is_reader=True, context=ctx, notify_address=addr
    )

    try:
        # Force the reader into persistent hot-path mode
        reader_cond.busy_loop_s = 9999
        reader_cond.record_read()

        # Writer sends a notification
        writer_cond.notify()

        # Give ZMQ a moment to deliver the inproc message
        time.sleep(0.05)

        # Reader calls wait() — without the fix this returns immediately
        # without ever seeing the notify.  With the fix it drains the
        # notification via _poll_notifications().
        reader_cond.wait(timeout_ms=0)

        # Verify the notify was consumed: a second poll on the socket
        # should return nothing.
        events = dict(reader_cond.poller.poll(timeout=0))
        notify_key = reader_cond.local_notify_socket
        assert notify_key not in events, (
            "Notify was not consumed by wait() in hot-path"
        )
    finally:
        for sock in (
            writer_cond.local_notify_socket,
            reader_cond.local_notify_socket,
            reader_cond.read_cancel_socket,
            reader_cond.write_cancel_socket,
        ):
            sock.close(linger=0)
        ctx.term()


def test_message_queue_notify_loss_no_deadlock():
    """Test that the message queue recovers gracefully when a reader
    experiences notify loss while in the hot-path.

    The scenario:
      1. Writer fills the ring buffer
      2. Reader stays in hot-path the whole time
      3. Writer wraps around and enters acquire_write waiting for ACKs
      4. The enqueue -> notify() call happens while reader is in hot-path
      5. Without the fix, the notify is lost and the reader blocks on the
         cold-path poller until SHM_READER_RECHECK_INTERVAL_MS (5s) elapses.
         With the fix, the notify is caught during the hot-path poll.
    """
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024,
        max_chunks=2,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()

    try:
        # Force reader into persistent hot-path
        reader._spin_condition.busy_loop_s = 9999

        # Fill the ring (writer writes max_chunks blocks)
        for i in range(writer.buffer.max_chunks):
            writer.enqueue({"seq": i})

        # Reader consumes all — this puts it in hot-path
        for i in range(writer.buffer.max_chunks):
            msg = reader.dequeue(timeout=5)
            assert msg["seq"] == i

        # Now the writer wraps the ring: the next enqueue triggers
        # acquire_write which waits for the reader's ACK.  The notify
        # is sent while the reader is in hot-path (busy_loop_s >> 0).
        writer.enqueue({"seq": 100})

        # Without the fix, this dequeue blocks for up to 5s
        # (SHM_READER_RECHECK_INTERVAL_MS) before seeing the data.
        # With the fix, it returns almost immediately.
        start = time.monotonic()
        msg = reader.dequeue(timeout=10)
        elapsed = time.monotonic() - start
        assert msg["seq"] == 100

        # With the fix, the dequeue should complete well under the
        # SHM_READER_RECHECK_INTERVAL_MS (5s).  We allow 1s to account
        # for scheduling jitter.
        assert elapsed < 1.0, (
            f"Dequeue took {elapsed:.2f}s, expected <1.0s with fix. "
            "This suggests the notify was missed and the reader relied "
            "on the periodic SHM recheck timeout."
        )
    finally:
        writer.shutdown()
        reader.shutdown()
        for sock in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            sock.close(linger=0)