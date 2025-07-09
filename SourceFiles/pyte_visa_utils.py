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
pyte_visa_utils
Tabor-Electronics SCPI based communication utilities based on `pyvisa`.
'''

from builtins import input
import sys
import socket
import warnings
import pyvisa as visa
import pyvisa.constants as vc

__version__ = '1.0.1'
__docformat__ = 'reStructuredText'

__all__ = [
    'open_session',
    'send_cmd']

 
def _list_udp_awg_instruments():
    '''Using UDP list all AWG-Instruments with LAN Interface.
    :returns: two lists: 1. VISA-Resource-Names 2. Instrument-IDN-Strings
    '''
    BROADCAST = '255.255.255.255'
    UDPSRVPORT = 7501
    UPFRMPORT = 7502
    FRMHEADERLEN = 22
    FRMDATALEN = 1024
    FLASHLINELEN = 32
    # FLASHOPCODELEN = 1

    vi_tcpip_resource_descs = []

    query_msg = bytearray([0xff] * FRMHEADERLEN)
  
    query_msg[0] = 0x54 
    query_msg[1] = 0x45 
    query_msg[2] = 0x49 
    query_msg[3] = 0x44 

    try:
        udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_IP)
        udp_server_sock.bind(("0.0.0.0", UDPSRVPORT))  # any IP-Address
        udp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, FRMHEADERLEN + FRMDATALEN)
        udp_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, True)

        # Send the query-message (to all)
        udp_server_sock.sendto(query_msg, (BROADCAST, UPFRMPORT))

        # Receive responses
        udp_server_sock.settimeout(2)
        while True:
            try:
                data, addr = \
                    udp_server_sock.recvfrom(FRMHEADERLEN + FRMDATALEN)

                ii = FRMHEADERLEN
                manuf_name = ''
                model_name = ''              
                address = ''
                port = ''               
                serial_nb = ''

                while ii + FLASHLINELEN <= len(data):
                    opcode = data[ii]
                    attr = data[ii + 1: ii + FLASHLINELEN - 1]
                    attr.rstrip()
                    if opcode == 0x44:
                        manuf_name = attr.decode('ascii').strip().strip('\x00')
                    elif opcode == 0x49:
                        model_name = attr.decode('ascii').strip().strip('\x00')
                    elif opcode == 0x53:
                        serial_nb = attr.decode('ascii').strip().strip('\x00')                   
                    elif opcode == 0x57:
                        address = attr.decode('ascii').strip().strip('\x00')                         
                    elif opcode == 0x51:
                        port = attr.decode('ascii').strip().strip('\x00') 
                        
                    ii = ii + FLASHLINELEN

                idn = '{0:s},{1:s},{2:s},{3:s},{4:s}'.format(manuf_name, model_name, serial_nb, address, port)
                vi_tcpip_resource_descs.append(idn)

            except socket.timeout:
                break
    except (IndexError, KeyError, NameError, AttributeError):
        pass

    return vi_tcpip_resource_descs


def _select_visa_rsc_name(
        rsc_manager=None,
        title=None,
        interface_name=None,
        use_ni_visa=True):
    """Select VISA Resource name.

    The supported interfaces names are: 'TCPIP', 'USB', 'GPIB', 'VXI', 'ASRL'

    If `use_ni_visa` is `False` then the pure Python
    back-end: 'PyVISA-py' will be used instead of NI-VISA
    (see https://pyvisa-py.readthedocs.io/en/latest/)

    :param rsc_manager: (optional) visa resource-manager.
    :param title: (optional) string displayed as title.
    :param interface_name: (optional) visa interface name.
    :param use_ni_visa: should use NI-VISA ? (`True` or `False`).
    :returns: the selected resource name (string).
    """

    if rsc_manager is None:
        if use_ni_visa:
            rsc_manager = visa.ResourceManager()
        else:
            rsc_manager = visa.ResourceManager('@py')

    selected_rsc_name = None

    rsc_names = []
    rsc_descs = []
    num_rscs = 0

    intf_nb = 0
    if interface_name is not None:
        intf_map = {'TCPIP': 1, 'USB': 2, 'GPIB': 3, 'VXI': 4, 'ASRL': 5}
        intf_nb = intf_map.get(interface_name, 0)

    while True:
        # uit_flag = True
        rsc_names = []
        rsc_descs = []
        num_rscs = 0

        if intf_nb in (1, 2, 3, 4, 5):
            choice = intf_nb
        else:
            if title is not None:
                print()
                print(title)
                print('=' * len(title))
            print()
            print("Select VISA Interface type:")
            print(" 1. TCPIP")
            print(" 2. USB")
            print(" 3. GPIB")
            print(" 4. VXI")
            print(" 5. ASRL")
            print(" 6. LXI")
            print(" 7. Enter VISA Resource-Name")
            print(" 8. Quit")
            choice = prompt_msg("Please enter your choice [1:7]: ", "123467")
            try:
                choice = int(choice)
            except ValueError:
                choice = -1
            print()

        if choice == 1:
            print()
            ip_str = prompt_msg(
                "Enter IP-Address, or press[Enter] to search:  ",)
            print()
            if len(ip_str) == 0:
                print('Searching AWG-Instruments ... ')
                rsc_names, rsc_descs = _list_udp_awg_instruments()
                print()
            else:
                try:
                    packed_ip = socket.inet_aton(ip_str)
                    ip_str = socket.inet_ntoa(packed_ip)
                    selected_rsc_name = \
                        "TCPIP::{0}::5025::SOCKET".format(ip_str)
                    break
                except OSError:
                    print()
                    print("Invalid IP-Address")
                    print()
                    continue
        elif choice == 2:
            rsc_names = rsc_manager.list_resources(query="?*USB?*INSTR")
        elif choice == 3:
            rsc_names = rsc_manager.list_resources(query="?*GPIB?*INSTR")
        elif choice == 4:
            rsc_names = rsc_manager.list_resources(query="?*VXI?*INSTR")
        elif choice == 5:
            rsc_names = rsc_manager.list_resources(query="?*ASRL?*INSTR")
        elif choice == 6:
            host_name = prompt_msg('Please enter Host-Name: ')
            if len(host_name) > 0:
                selected_rsc_name = "TCPIP::{0}::INSTR".format(host_name)
                break
        elif choice == 7:
            resource_name = prompt_msg('Please enter VISA Resource-Name: ')
            print()
            if len(resource_name) > 0:
                selected_rsc_name = resource_name
                break
        elif choice == 8:
            break
        else:
            print()
            print("Invalid choice")
            print()
            continue

        num_rscs = len(rsc_names)
        if num_rscs == 0:
            print()
            print('No VISA Resource was found!')
            yes_no = prompt_msg("Do you want to retry [y/n]: ", "yYnN")
            if yes_no in "yY":
                continue
            else:
                break
        elif num_rscs == 1 and choice != 1:
            selected_rsc_name = rsc_names[0]
            break
        elif num_rscs > 1 or (num_rscs == 1 and choice == 1):
            if len(rsc_descs) != num_rscs:
                rsc_descs = ["" for n in range(num_rscs)]
                # get resources descriptions:
                for n, name in zip(range(num_rscs), rsc_names):
                    vi = None
                    try:
                        vi = rsc_manager.open_resource(name)
                        if vi is not None:
                            vi.read_termination = '\n'
                            vi.write_termination = '\n'
                            ans_str = vi.query('*IDN?')
                            rsc_descs[n] = ans_str
                    except (visa.Error, AttributeError, NameError, IndexError):
                        pass
                    if vi is not None:
                        try:
                            vi.close()
                            vi = None
                        except visa.Error:
                            pass

            print("Please choose one of the available devices:")
            for n, name, desc in zip(range(num_rscs), rsc_names, rsc_descs):
                print(" {0:d}. {1} ({2})".format(n+1, desc, name))
            print(" {0:d}. Back to main menu".format(num_rscs+1))
            msg = "Please enter your choice [{0:d}:{1:d}]: "
            msg = msg.format(1, num_rscs + 1)
            valid_answers = [str(i+1) for i in range(num_rscs+1)]
            choice = prompt_msg(msg, valid_answers)

            try:
                choice = int(choice)
            except ValueError:
                choice = num_rscs+1

            if choice == num_rscs+1:
                continue
            else:
                selected_rsc_name = rsc_names[choice - 1]
                break

    return selected_rsc_name


def _init_vi_inst(
        vi,
        timeout_msec=10000,
        read_buff_size_bytes=4096,
        write_buff_size_bytes=4096):
    '''Initialize the given Instrument VISA Session.

    :param vi: `pyvisa` instrument.
    :param timeout_msec: VISA-Timeout (in milliseconds)
    :param read_buff_size_bytes: VISA Read-Buffer Size (in bytes)
    :param write_buff_size_bytes: VISA Write-Buffer Size (in bytes)
    '''

    if vi is not None:
        vi.timeout = int(timeout_msec)
        try:
            vi.visalib.set_buffer(
                vi.session, vc.VI_READ_BUF, int(read_buff_size_bytes))
        except NotImplementedError:
            vi.set_visa_attribute(vc.VI_READ_BUF, int(read_buff_size_bytes))
        vi.__dict__['read_buff_size'] = read_buff_size_bytes
        try:
            vi.visalib.set_buffer(
                vi.session, vc.VI_WRITE_BUF, int(write_buff_size_bytes))
        except NotImplementedError:
            vi.set_visa_attribute(vc.VI_WRITE_BUF, int(write_buff_size_bytes))
        vi.__dict__['write_buff_size'] = write_buff_size_bytes
        vi.read_termination = '\n'
        vi.write_termination = '\n'
        intf_type = vi.get_visa_attribute(vc.VI_ATTR_INTF_TYPE)
        if intf_type in (vc.VI_INTF_USB,
                         vc.VI_INTF_GPIB,
                         vc.VI_INTF_TCPIP,
                         vc.VI_INTF_ASRL):
            vi.set_visa_attribute(
                vc.VI_ATTR_WR_BUF_OPER_MODE, vc.VI_FLUSH_ON_ACCESS)
            vi.set_visa_attribute(
                vc.VI_ATTR_RD_BUF_OPER_MODE, vc.VI_FLUSH_ON_ACCESS)
            if intf_type == vc.VI_INTF_TCPIP:
                vi.set_visa_attribute(
                    vc.VI_ATTR_TERMCHAR_EN, vc.VI_TRUE)   # vc.VI_FALSE
            elif intf_type == vc.VI_INTF_ASRL:
                vi.set_visa_attribute(vc.VI_ATTR_ASRL_BAUD, 115200)
                vi.set_visa_attribute(vc.VI_ATTR_ASRL_END_OUT, 0)
                vi.set_visa_attribute(vc.VI_ATTR_ASRL_END_IN, 2)
        vi.clear()


def open_session(
        resource_name=None,
        title_msg=None,
        vi_rsc_mgr=None,
        extra_init=True,
        use_ni_visa=True):
    '''Open VISA Session (optionally prompt for resource name).

    The `resource_name` can be either:
        1. Full VISA Resource-Name (e.g. 'TCPIP::192.168.0.170::5025::SOCKET')
        2. IP-Address (e.g. '192.168.0.170')
        3. Interface-Name (either 'TCPIP', 'USB', 'GPIB', 'VXI' or 'ASRL')
        4. None

    If `use_ni_visa` is `False` then the pure python
    backend: 'PyVISA-py' will be used instead of NI-VISA
    (see https://pyvisa-py.readthedocs.io/en/latest/)

    :param resource_name: the Resource-Name
    :param title_msg: title-message (for the interactive-menu)
    :param vi_rsc_mgr: VISA Resource-Manager
    :param extra_init: should perform extra initialization.
    :param use_ni_visa: should use NI-VISA ? (`True` or `False`).
    :returns: `pyvisa` instrument.

    Example:

        >>> import pyte
        >>>
        >>> # Connect to Arbitrary-Wave-Generator Instrument through TCPIP
        >>> # (the user will be asked to enter the instrument's IP-Address):
        >>> vi = pyte.open_session(
        >>>        resource_name='TCPIP',
        >>>        title_msg='Connect to AWG Instrument')
        >>>
        >>> # Connect to Digital-Multimeter through USB:
        >>> dmm = pyte.open_session(resource_name='USB', extra_init=False)
        >>>
        >>> print vi.query('*IDN?')
        >>> print dmm.query('*IDN?')
        >>>
        >>> # Do some work ..
        >>>
        >>> vi.close()
        >>> dmm.close()

    '''

    vi = None
    try:

        if vi_rsc_mgr is None:
            if use_ni_visa:
                vi_rsc_mgr = visa.ResourceManager()
            else:
                vi_rsc_mgr = visa.ResourceManager('@py')

        if resource_name is None:
            resource_name = _select_visa_rsc_name(vi_rsc_mgr, title_msg)
        elif resource_name.upper() in ('TCPIP', 'USB', 'GPIB', 'VXI', 'ASRL'):
            resource_name = _select_visa_rsc_name(
                vi_rsc_mgr, title_msg, resource_name.upper())
        else:
            try:
                packed_ip = socket.inet_aton(resource_name)
                ip_str = socket.inet_ntoa(packed_ip)
                if resource_name == ip_str:
                    resource_name = "TCPIP::{0}::5025::SOCKET".format(ip_str)
            except OSError:
                pass

        if resource_name is None:
            return None

        vi = vi_rsc_mgr.open_resource(resource_name)
        if extra_init and vi is not None:
            _init_vi_inst(vi)
    except visa.Error:
        err_msg = 'Failed to open "{0}"\n{1}'
        err_msg = err_msg.format(resource_name, sys.exc_info())
        print(err_msg)

    return vi


def prompt_msg(msg, valid_answers=None):
    """Prompt message and return user's answer."""
    ans = input(msg)
    if valid_answers is not None:
        count = 0
        while ans not in valid_answers:
            count += 1
            ans = input(msg)
            if count == 5:
                break
    return ans


def get_visa_err_desc(err_code):
    '''Get description of the given visa error code.'''
    desc = None
    try:
        from pyvisa.errors import completion_and_error_messages
        desc = completion_and_error_messages.get(err_code)
    except (ImportError, NameError, AttributeError):
        pass
    if desc is None:
        desc = 'VISA-Error {0:x}'.format(int(err_code))

    return desc


def send_cmd(vi, cmd_str, paranoia_level=1):
    '''Send (SCPI) Command to Instrument

    :param vi: `pyvisa` instrument.
    :param cmd_str: the command string.
    :param paranoia_level: paranoia-level (0:low, 1:normal, 2:high)
    '''
    if paranoia_level == 1:
        ask_str = cmd_str.rstrip()
        if len(ask_str) > 0:
            ask_str += '; *OPC?'
        else:
            ask_str = '*OPC?'
        _ = vi.query(ask_str)
    elif paranoia_level >= 2:
        ask_str = cmd_str.rstrip()
        if len(ask_str) > 0:
            ask_str += '; :SYST:ERR?'
        else:
            ask_str = ':SYST:ERR?'
        syst_err = vi.query(ask_str)
        try:
            errnb = int(syst_err.split(',')[0])
        except visa.VisaIOError:
            errnb = -1
        if errnb != 0:
            syst_err = syst_err.rstrip()
            wrn_msg = 'ERR: "{0}" after CMD: "{1}"'.format(syst_err, cmd_str)
            _ = vi.query('*CLS; *OPC?')  # clear the error-list
            if paranoia_level >= 3:
                raise NameError(wrn_msg)
            else:
                warnings.warn(wrn_msg)

    else:
        vi.write(cmd_str)
        # vi.visalib.viFlush(vi.session, vc.VI_WRITE_BUF)
