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

"""
Quick n dirty tbb detection
"""

from waflib.Configure import conf

def options(opt):
    opt.add_option('--tbb', type='string', help='path to Intel TBB', dest='tbb')

# check if a lib exists for both osx (darwin) and GNU/linux
def check_lib(self, name, path):
    if self.env['DEST_OS'] == 'darwin':
        libname = name + '.dylib'
    else:
        libname = name + '.so'
    res = self.find_file(libname, path)
    lib = res[:-len(libname)-1]
    return lib

@conf
def check_tbb(self, *k, **kw):
    def get_directory(filename, dirs):
        res = self.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    if self.options.tbb:
        includes_tbb = [self.options.tbb + '/include']
        libpath_tbb = [self.options.tbb + '/lib', self.options.tbb + '/lib64']
    else:
        includes_tbb = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include', '/opt/homebrew/include']
        libpath_tbb = ['/usr/lib', '/usr/local/lib64', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/', '/usr/local/lib/x86_64-linux-gnu/', '/usr/lib/aarch64-linux-gnu/', '/usr/local/lib/aarch64-linux-gnu/', '/opt/homebrew/lib']

    self.start_msg('Checking Intel TBB includes')
    incl = ''
    lib = ''
    try:
        incl = get_directory('tbb/parallel_for.h', includes_tbb)
        self.end_msg(incl)
    except:
        if required:
            self.fatal('Not found in %s' % str(includes_tbb))
        self.end_msg('Not found in %s' % str(includes_tbb), 'YELLOW')
        return

    # check for oneapi vs older tbb
    self.start_msg('Checking for Intel OneAPI TBB')
    using_oneapi = False
    try:
        incl = get_directory('oneapi/tbb.h', includes_tbb)
        self.end_msg(incl)
        using_oneapi = True
    except:
        self.end_msg('Not found in %s, reverting to older TBB' % str(includes_tbb), 'YELLOW')
        using_oneapi = False


    self.start_msg('Checking Intel TBB libs')
    try:
        lib = check_lib(self, 'libtbb', libpath_tbb)
        self.end_msg(lib)
    except:
        if required:
            self.fatal('Not found in %s' % str(libpath_tbb))
        self.end_msg('Not found in %s' % str(libpath_tbb), 'YELLOW')
        return

    self.env.LIBPATH_TBB = [lib]
    self.env.LIB_TBB = ['tbb']
    self.env.INCLUDES_TBB = [incl]
    self.env.DEFINES_TBB = ['USE_TBB']
    if using_oneapi:
        self.env.DEFINES_TBB += ['USE_TBB_ONEAPI']
