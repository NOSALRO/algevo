#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2025 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2023-2025 Laboratory of Automation and Robotics, University of Patras, Greece
#|    Copyright (c) 2022-2025 Konstantinos Chatzilygeroudis
#|    Authors:  Konstantinos Chatzilygeroudis
#|    email:    costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              https://lar.upatras.gr/
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
Quick n dirty ifopt detection
"""

from waflib.Configure import conf

def options(opt):
    opt.add_option('--ifopt', type='string', help='path to ifopt', dest='ifopt')

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
def check_ifopt(self, *k, **kw):
    def get_directory(filename, dirs):
        res = self.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    if self.options.ifopt:
        includes_ifopt = [self.options.ifopt + '/include']
        libpath_ifopt = [self.options.ifopt + '/lib']
    else:
        includes_ifopt = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include', '/opt/homebrew/include']
        libpath_ifopt = ['/usr/lib', '/usr/local/lib64', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/', '/usr/local/lib/x86_64-linux-gnu/', '/usr/lib/aarch64-linux-gnu/', '/usr/local/lib/aarch64-linux-gnu/', '/opt/homebrew/lib']

    self.start_msg('Checking ifopt includes')
    incl = ''
    lib = ''
    try:
        incl = get_directory('ifopt/problem.h', includes_ifopt)
        self.end_msg(incl)
    except:
        if required:
            self.fatal('Not found in %s' % str(includes_ifopt))
        self.end_msg('Not found in %s' % str(includes_ifopt), 'YELLOW')
        return

    self.start_msg('Checking ifopt libs')
    libs = ['ifopt_core', 'ifopt_ipopt']
    paths = []
    for lib in libs:
        try:
            lib_path = check_lib(self, 'lib' + lib, libpath_ifopt)
            paths.append(lib_path)
        except:
            if required:
                self.fatal('Not found in %s' % str(libpath_ifopt))
            self.end_msg('Not found in %s' % str(libpath_ifopt), 'YELLOW')
            return
    self.end_msg(list(set(paths)))

    self.env.LIBPATH_IFOPT = list(set(paths))
    self.env.LIB_IFOPT = libs
    self.env.INCLUDES_IFOPT = [incl]
