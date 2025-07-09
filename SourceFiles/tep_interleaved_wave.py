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
tep_interleaved_wave - functions to create interleaved-wave for Proteus.
'''

import numpy as np

__version__ = '1.0.1'
__docformat__ = 'reStructuredText'

__all__ = [
    'compose_interleaved_two',
    'untie_interleaved_two',
    'compose_interleaved_four',
    'untie_interleaved_four']


def compose_interleaved_two(wave1, wave2):
    '''
    Compose a two-dimensional interleaved waveform
    from the given two (one-dimensional) waveforms.

    :param wave1: the first wave (a `numpy` array of `uint16` items).
    :param wave2: the second wave (a `numpy` array of `uint16` items).
    :returns: 2-dimensional interleaved wave (`numpy` array of `uint16` items).
    '''

    assert (len(wave1) == len(wave2))

    half_len = len(wave1)

    a = np.empty(2 * half_len, dtype=np.uint16)
    b = a.view()
    b.shape = (half_len, 2)
    b[:, 0] = wave1
    b[:, 1] = wave2
    return a


def untie_interleaved_two(w):
    '''
    Untie a two-dimensional interleaved waveform
    into two one-dimensional waveforms.

    :param w: 2-dimensional interleaved wave (`numpy` array of `uint16` items).
    :returns: two 1-dimensional waves. (2 `numpy` arrays of `uint16` items).
    '''

    assert (len(w) % 2 == 0)

    half_len = len(w) // 2

    w1 = np.empty(half_len, dtype=np.uint16)
    w2 = np.empty(half_len, dtype=np.uint16)
    a = w.view()
    a.shape = (half_len, 2)
    w1[:] = a[:, 0]
    w2[:] = a[:, 1]
    return w1, w2


def compose_interleaved_four(w1, w2, w3, w4):
    '''
    Compose a four-dimensional interleaved waveform
    from the given four (one-dimensional) waveforms.

    :param w1: the first wave (a `numpy` array of `uint16` items).
    :param w2: the second wave (a `numpy` array of `uint16` items).
    :param w3: the third wave (a `numpy` array of `uint16` items).
    :param w4: the fourth wave (a `numpy` array of `uint16` items).
    :returns: 2-dimensional interleaved wave (`numpy` array of `uint16` items).
    '''

    assert (len(w1) == len(w2) == len(w3) == len(w4))

    fourth_len = len(w1)

    w = np.empty(4 * fourth_len, dtype=np.uint16)

    a = w.view(np.uint8)
    a1 = w1.view(np.uint8)
    a2 = w2.view(np.uint8)
    a3 = w3.view(np.uint8)
    a4 = w4.view(np.uint8)

    for n in range(fourth_len):
        a[n * 8] = a1[2 * n]
        a[n * 8 + 1] = a2[2 * n]
        a[n * 8 + 2] = a3[2 * n]
        a[n * 8 + 3] = a4[2 * n]

        a[n * 8 + 4] = a1[2 * n + 1]
        a[n * 8 + 5] = a2[2 * n + 1]
        a[n * 8 + 6] = a3[2 * n + 1]
        a[n * 8 + 7] = a4[2 * n + 1]

    return w


def untie_interleaved_four(w):
    '''
    Untie a four-dimensional interleaved waveform
    into four one-dimensional waveforms.

    :param w: 4-dimensional interleaved wave (`numpy` array of `uint16` items).
    :returns: four 1-dimensional waves. (4 `numpy` arrays of `uint16` items).
    '''

    assert (len(w) % 4 == 0)

    fourth_len = len(w) // 4

    w1 = np.empty(fourth_len, dtype=np.uint16)
    w2 = np.empty(fourth_len, dtype=np.uint16)
    w3 = np.empty(fourth_len, dtype=np.uint16)
    w4 = np.empty(fourth_len, dtype=np.uint16)

    a = w.view(np.uint8)
    a1 = w1.view(np.uint8)
    a2 = w2.view(np.uint8)
    a3 = w3.view(np.uint8)
    a4 = w4.view(np.uint8)

    for n in range(fourth_len):
        a1[2 * n] = a[n * 8]
        a2[2 * n] = a[n * 8 + 1]
        a3[2 * n] = a[n * 8 + 2]
        a4[2 * n] = a[n * 8 + 3]

        a1[2 * n + 1] = a[n * 8 + 4]
        a2[2 * n + 1] = a[n * 8 + 5]
        a3[2 * n + 1] = a[n * 8 + 6]
        a4[2 * n + 1] = a[n * 8 + 7]

    return w1, w2, w3, w4
