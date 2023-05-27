#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Konstantinos Chatzilygeroudis
#|    email:    costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of algevo.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
#! /usr/bin/env python
# Konstantinos Chatzilygeroudis - 2022

"""
Quick n dirty proxqp detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
    opt.add_option('--proxqp', type='string', help='path to proxqp', dest='proxqp')

@conf
def check_proxqp(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]
    includes_check = ['/usr/include', '/usr/local/include', '/opt/include', '/opt/local/include', '/opt/proxsuite/include', '/sw/include', '/opt/homebrew/include']

    required = kw.get('required', False)

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.proxqp:
        includes_check = [conf.options.proxqp]

    try:
        conf.start_msg('Checking for ProxQP')
        incl = get_directory('proxsuite/proxqp/results.hpp', includes_check)
        conf.env.INCLUDES_PROXQP = [incl]
        conf.end_msg(incl)
    except:
        if required:
            conf.fatal('Not found in %s' % str(includes_check))
        conf.end_msg('Not found in %s' % str(includes_check), 'RED')
        return

    return
