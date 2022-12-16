#!/usr/bin/env python
# encoding: utf-8
#| Konstantinos Chatzilygeroudis 2022

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
