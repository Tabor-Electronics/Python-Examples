#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2021 Tabor Electronics Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
tep_task_table - Proteus task-table utilities.

The class :class:`tep_task_table.TaskTableRow` represents a single row
in the task-table of Proteus instruments, where each column (field) is
represented by property.

The methods
 - :meth:`tep_task_table.TaskTableRow.pack`
 - :meth:`tep_task_table.TaskTableRow.unpack`

can be used for packing / unpacking the fields
to (or from) binary-format.
'''

from enum import Enum
import numpy as np
import struct

__version__ = '1.0.1'
__docformat__ = 'reStructuredText'

__all__ = [
    'TaskType',
    'TaskIdleWav',
    'TaskEnableAbort',
    'TaskJumpMode',
    'TaskDestSel',
    'TaskTableRow']


class TaskType(Enum):
    '''Enumerates task types.'''

    # Single task (not part of a task-sequence)
    SINGLE = 0

    # First task in task-sequence.
    START = 1

    # Last task in task-sequence.
    END = 2

    # Inside a task-sequence.
    SEQ = 3

    def __str__(self):
        return str(self.name)


class TaskIdleWav(Enum):
    '''Enumerates task idle-waveform types.'''

    # DC waveform with the task's DC value.
    DC = 0

    # DC waveform with the current segment's first point.
    FIRST = 1

    # Waveform with the current segment's data.
    CURRUNT = 2

    def __str__(self):
        return str(self.name)


class TaskEnableAbort(Enum):
    '''Enumerates task's enable/abort signal types.'''

    # None (Continuous).
    NONE = 0

    # External Trigger 1.
    TRIG1 = 1

    # External Trigger 2.
    TRIG2 = 2

    # Internal Trigger.
    INTERN = 3

    # CPU Trigger
    CPU = 4

    # Feedback Trigger.
    FBTRG = 5

    # Hardware Control Pins.
    HWC = 6

    def __str__(self):
        return str(self.name)


class TaskJumpMode(Enum):
    '''Enumerates task jumping modes (upon abort signal).'''

    # Finish the current task and jump.
    EVENTUALLY = 0

    # Jump immediately.
    IMMEDIATE = 1

    def __str__(self):
        return str(self.name)


class TaskDestSel(Enum):
    '''Enumerates task modes for selecting next-task destination.'''

    # Go to the task defined by the field `next_task1`.
    NEXT = 0

    # Select the next task by the feedback trigger value.
    FBTRG = 1

    # Go to `next_task1` upon trigger1 and to `next_task2` upon trigger2.
    TRGSEL = 2

    # Go to next task in the task-table.
    NTSEL = 3

    # Go to the first task in the next scenario.
    SCEN = 4

    # Go to `next_task1` if the digitizer-signal is high
    # and to `next_task2` if the digitizer-signal is low.
    DSIG = 5

    def __str__(self):
        return str(self.name)


class TaskTableRow(object):
    '''
    Represents a single row (entry) in the task-table of Proteus instruments.
    Each column in the task-table is represented by a corresponding property in
    this class.

    The methods
     - :meth:`TaskTableRow.pack()`
     - :meth:`TaskTableRow.unpack()`

    packs (unpacks) the fields to (from) binary-format.
    '''

    @classmethod
    def columns(cls):
        '''Gets the column names of the task-table.'''
        return[
            'task_type',
            'seg_num',
            'next_task1',
            'next_task2',
            'task_loops',
            'seq_loops',
            'idle_wave',
            'idle_dc_level',
            'enable_signal',
            'abort_signal',
            'jump_mode',
            'dest_sel',
            'delay_ticks',
            'keep_loop_trig',
            'trig_digitizer']

    def __init__(
            self,
            task_type=TaskType.SINGLE,
            seg_num=1,
            next_task1=1,
            next_task2=0,
            task_loops=1,
            seq_loops=1,
            idle_wave=TaskIdleWav.DC,
            idle_dc_level=0,
            enable_signal=TaskEnableAbort.NONE,
            abort_signal=TaskEnableAbort.NONE,
            jump_mode=TaskJumpMode.EVENTUALLY,
            dest_sel=TaskDestSel.NEXT,
            delay_ticks=0,
            keep_loop_trig=False,
            trig_digitizer=False):
        '''Constructor.

        :param task_type: the type of this task (`TaskType`).
        :param seg_num: the segment-number that is associated with this task.
        :param next_task1: the task-number of next-task 1 (zero for end).
        :param next_task2: the task-number of next-task 2 (zero for end).
        :param task_loops: the number of task-loops (maximum: 2**20 - 1).
        :param seq_loops: the number of sequence-loops (maximum: 2**20 - 1).
        :param idle_wave: the waveform during task idle-time (`TaskIdleWav`).
        :param idle_dc_level: the DAC level of idle DC waveform (0 to 65535).
        :param enable_signal: the task's enabling signal (`TaskEnableAbort`).
        :param abort_signal: the task's aborting signal (`TaskEnableAbort`).
        :param jump_mode: the jumping-mode upon abort-signal (`TaskJumpMode`).
        :param dest_sel: select-mode of next-task destination (`TaskDestSel`).
        :param delay_ticks: delay in clocks before leaving task (0 to 65535).
        :param keep_loop_trig: should keep waiting to trigger on each loop?
        :param trig_digitizer: should generate digitizer-trigger at task end?
        '''

        self._task_type = TaskType(task_type)
        self._seg_num = np.uint32(seg_num)
        self._next_task1 = np.uint32(next_task1)
        self._next_task2 = np.uint32(next_task2)
        self._task_loops = np.uint32(task_loops)
        self._seq_loops = np.uint32(seq_loops)
        self._idle_wave = TaskIdleWav(idle_wave)
        self._idle_dc_level = np.uint16(idle_dc_level)
        self._enable_signal = TaskEnableAbort(enable_signal)
        self._abort_signal = TaskEnableAbort(abort_signal)
        self._jump_mode = TaskJumpMode(jump_mode)
        self._dest_sel = TaskDestSel(dest_sel)
        self._delay_ticks = np.uint16(delay_ticks)
        self._keep_loop_trig = bool(keep_loop_trig)
        self._trig_digitizer = bool(trig_digitizer)

    def __iter__(self):
        '''Get iterator over the row's fields (columns).'''
        yield self._task_type
        yield self._seg_num
        yield self._next_task1
        yield self._next_task2
        yield self._task_loops
        yield self._seq_loops
        yield self._idle_wave
        yield self._idle_dc_level
        yield self._enable_signal
        yield self._abort_signal
        yield self._jump_mode
        yield self._dest_sel
        yield self._delay_ticks
        yield self._keep_loop_trig
        yield self._trig_digitizer

    @staticmethod
    def row_size():
        '''
        Gets the size in bytes of single (serialized) task-table row.
        :return: the size in bytes of single (serialized) task-table row.
        '''
        return 32

    @property
    def task_type(self):
        '''
        The type of this task.
        :getter: Gets the type of this task.
        :setter: Sets the type of this task.
        :type: `TaskType`.
        '''
        return self._task_type

    @task_type.setter
    def task_type(self, value):
        self._task_type = TaskType(value)

    @property
    def seg_num(self):
        '''
        The segment-number that is associated with this task.
        :getter: Gets the segment-number that is associated with this task.
        :setter: Sets segment-number that is associated with this task.
        :type: `numpy.uint32`.
        '''
        return self._seg_num

    @seg_num.setter
    def seg_num(self, value):
        self._seg_num = np.uint32(value)

    @property
    def next_task1(self):
        '''
        The task-number of next-task number 1 (zero for end).
        :getter: Gets the task-number of next-task number 1.
        :setter: Sets the task-number of next-task number 1.
        :type: `numpy.uint32`.
        '''
        return self._next_task1

    @next_task1.setter
    def next_task1(self, value):
        self._next_task1 = np.uint32(value)

    @property
    def next_task2(self):
        '''
        The task-number of next-task number 2 (zero for end).
        :getter: Gets the task-number of next-task number 2.
        :setter: Sets the task-number of next-task number 2.
        :type: `numpy.uint32`.
        '''
        return self._next_task2

    @next_task2.setter
    def next_task2(self, value):
        self._next_task2 = np.uint32(value)

    @property
    def task_loops(self):
        '''
        The number of loops over the associated segment (maximum: 2**20 - 1).
        :getter: Gets the number of loops over the associated segment.
        :setter: Sets the number of loops over the associated segment.
        :type: `numpy.uint32`.
        '''
        return self._task_loops

    @task_loops.setter
    def task_loops(self, value):
        self._task_loops = np.uint32(value)

    @property
    def seq_loops(self):
        '''
        The number of loops over the tasks-sequence (maximum: 2**20 - 1).
        :getter: Gets the number of loops over the tasks-sequence.
        :setter: Sets the number of loops over the tasks-sequence.
        :type: `numpy.uint32`.
        '''
        return self._seq_loops

    @seq_loops.setter
    def seq_loops(self, value):
        self._seq_loops = np.uint32(value)

    @property
    def idle_wave(self):
        '''
        The type of waveform that played when the task is idle.
        :getter: Gets the type of waveform that played when the task is idle.
        :setter: Sets type of waveform that played when the task is idle.
        :type: `TaskIdleWav`.
        '''
        return self._idle_wave

    @idle_wave.setter
    def idle_wave(self, value):
        self._idle_wave = TaskIdleWav(value)

    @property
    def idle_dc_level(self):
        '''
        The DAC level (between 0 and 65535) of the *idle DC waveform*.
        :getter: Gets the DAC level of the *idle DC waveform*.
        :setter: Sets the DAC level of the *idle DC waveform*.
        :type: `numpy.uint16`.
        '''
        return self._idle_dc_level

    @idle_dc_level.setter
    def idle_dc_level(self, value):
        self._idle_dc_level = np.uint16(value)

    @property
    def enable_signal(self):
        '''
        The task's enabling signal.
        :getter: Gets the task's enabling signal.
        :setter: Sets the task's enabling signal.
        :type: `TaskEnableAbort`.
        '''
        return self._enable_signal

    @enable_signal.setter
    def enable_signal(self, value):
        self._enable_signal = TaskEnableAbort(value)

    @property
    def abort_signal(self):
        '''
        The task's aborting signal.
        :getter: Gets the task's aborting signal.
        :setter: Sets the task's aborting signal.
        :type: `TaskEnableAbort`.
        '''
        return self._abort_signal

    @abort_signal.setter
    def abort_signal(self, value):
        self._abort_signal = TaskEnableAbort(value)

    @property
    def jump_mode(self):
        '''
        The jumping-mode (eventually or immediately) upon abort-signal
        :getter: Gets the task's jumping-mode.
        :setter: Sets the task's jumping-mode.
        :type: `TaskJumpMode`.
        '''
        return self._jump_mode

    @jump_mode.setter
    def jump_mode(self, value):
        self._jump_mode = TaskJumpMode(value)

    @property
    def dest_sel(self):
        '''
        The selecting-mode of next-task destination.
        :getter: Gets the selecting-mode of next-task destination.
        :setter: Sets the selecting-mode of next-task destination.
        :type: `TaskDestSel`.
        '''
        return self._dest_sel

    @dest_sel.setter
    def dest_sel(self, value):
        self._dest_sel = TaskDestSel(value)

    @property
    def delay_ticks(self):
        '''
        The delay in clock-ticks before leaving the task (maximum: 65535).
        :getter: Gets the delay in clock-ticks before leaving the task.
        :setter: Sets the delay in clock-ticks before leaving the task.
        :type: `numpy.uint16`.
        '''
        return self._delay_ticks

    @delay_ticks.setter
    def delay_ticks(self, value):
        self._delay_ticks = np.uint16(value)

    @property
    def keep_loop_trig(self):
        '''
        Indicates whether should keep waiting to trigger on each loop.
        :getter: Gets whether should keep waiting to trigger on each loop.
        :setter: Sets  whether should keep waiting to trigger on each loop.
        :type: boolean.
        '''
        return self._keep_loop_trig

    @keep_loop_trig.setter
    def keep_loop_trig(self, value):
        self._keep_loop_trig = bool(value)

    @property
    def trig_digitizer(self):
        '''
        Indicates whether should trigger the digitizer at the end of the task.
        :getter: Gets whether should trigger the digitizer at end of task.
        :setter: Sets whether should trigger the digitizer at end of task.
        :type: boolean.
        '''
        return self._trig_digitizer

    @trig_digitizer.setter
    def trig_digitizer(self, value):
        self._trig_digitizer = bool(value)

    # The arrangement of the paced bytes array
    _packed_bytes_arrangement = struct.Struct(
        ' '.join((
            '<',  # little-endian bytes-order
            'L',  # segNb (u32)
            'L',  # nextTask1 (u32)
            'L',  # nextTask2 (u32)
            'L',  # taskLoopCount (u32)
            'L',  # seqLoopCount (u32)
            'H',  # nextTaskDelay (u16)
            'H',  # taskDcVal (u16)
            'B',  # taskIdlWvf (u8)
            'B',  # taskEnableSig (u8)
            'B',  # taskAbortSig (u8)
            'B',  # taskCondJumpSel (u8)
            'B',  # taskAbortJumpType (u8)
            'B',  # taskState (u8)
            'B',  # taskLoopTrigEn (u8)
            'B',  # genAdcTrigger (u8)
            )))

    def pack(self, byte_array=None, byte_array_offs=0):
        '''
        Pack this task-table row into byte-array,
        in the binary-format the instrument expects.

        :param byte_array: a `numpy` array to pack into (optional).
        :param byte_array_offs: the offset in `byte_array` (optional)
        :returns: a `numpy` array of 32 `uint8` items.
        '''
        if byte_array is None:
            byte_array_offs = 0
            byte_array = np.empty(32, dtype=np.uint8)

        self._packed_bytes_arrangement.pack_into(
            byte_array,
            byte_array_offs,
            np.uint32(self._seg_num),
            np.uint32(self._next_task1),
            np.uint32(self._next_task2),
            np.uint32(self._task_loops),
            np.uint32(self._seq_loops),
            np.uint16(self._delay_ticks),
            np.uint16(self._idle_dc_level),
            np.uint8(self._idle_wave.value),
            np.uint8(self._enable_signal.value),
            np.uint8(self._abort_signal.value),
            np.uint8(self._dest_sel.value),
            np.uint8(self._jump_mode.value),
            np.uint8(self._task_type.value),
            np.uint8(self._keep_loop_trig),
            np.uint8(self._trig_digitizer))

        return byte_array[byte_array_offs: byte_array_offs + 32]

    def unpack(self, byte_array, byte_array_offs=0):
        '''
        Unpack the fields of this task-table row from byte-array,
        in the binary-format the instrument uses.

        :param byte_array: a `numpy` array to unpack from.
        :param byte_array_offs: the offset in `byte_array` (optional)
        :returns: None.
        '''

        fields = self._packed_bytes_arrangement.unpack_from(
            byte_array,
            byte_array_offs)

        self._seg_num = np.uint32(fields[0])
        self._next_task1 = np.uint32(fields[1])
        self._next_task2 = np.uint32(fields[2])
        self._task_loops = np.uint32(fields[3])
        self._seq_loops = np.uint32(fields[4])
        self._delay_ticks = np.uint16(fields[5])
        self._idle_dc_level = np.uint16(fields[6])
        self._idle_wave = TaskIdleWav(fields[7])
        self._enable_signal = TaskEnableAbort(fields[8])
        self._abort_signal = TaskEnableAbort(fields[9])
        self._dest_sel = TaskDestSel(fields[10])
        self._jump_mode = TaskJumpMode(fields[11])
        self._task_type = TaskType(fields[12])
        self._keep_loop_trig = bool(fields[13])
        self._trig_digitizer = bool(fields[14])
