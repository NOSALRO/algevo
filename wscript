#!/usr/bin/env python
# encoding: utf-8
import sys
import os
import fnmatch
import glob
sys.path.insert(0, sys.path[0]+'/waf_tools')

VERSION = '1.0.0'
APPNAME = 'algevo'

srcdir = '.'
blddir = 'build'

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test
import avx


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('eigen')
    opt.load('tbb')
    opt.load('proxqp')
    opt.load('simde')
    opt.load('ifopt')
    opt.load('towr')

    opt.add_option('--no-native', action='store_true', help='Do not compile with march=native optimization flags', dest='no_native')
    opt.add_option('--exp', type='string',
                   help='exp(s) to build, separate by comma', dest='exp')

    for i in glob.glob('exp/*'):
        if os.path.isdir(i):
            opt.start_msg('command-line options for [%s]' % i)
            try:
                opt.recurse(i)
                opt.end_msg(' -> OK')
            except WafError:
                opt.end_msg(' -> no options found')

def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('eigen')
    conf.load('tbb')
    conf.load('proxqp')
    conf.load('simde')
    conf.load('ifopt')
    conf.load('towr')

    conf.check_eigen(required=True)
    conf.check_tbb(required=True)
    conf.check_proxqp(required=False)
    conf.check_simde(required=False)
    conf.check_ifopt(required=False)
    conf.check_towr(required=False)

    conf.check(features='cxx cxxprogram', lib=['pthread'], uselib_store='PTHREAD')

    native = '-march=native -DPROXSUITE_VECTORIZE'
    native_icc = 'mtune=native -DPROXSUITE_VECTORIZE'

    if conf.options.no_native:
        native = ''
        native_icc = ''
    elif 'INCLUDES_SIMDE' not in conf.env or 'INCLUDES_PROXQP' not in conf.env:
        native = '-march=native'
        native_icc = 'mtune=native'

    # We require C++17
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -xHost -unroll -g " + native_icc
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++17"
        # no-stack-check required for Catalina
        opt_flags = " -O3 -g -faligned-new -fno-stack-check -Wno-narrowing" + native
    else:
        gcc_version = int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1])
        if gcc_version < 50:
            conf.fatal('We need C++17 features. Your compiler does not support them!')
        else:
            common_flags = "-Wall -std=c++17"
        opt_flags = " -O3 -g " + native
        if gcc_version >= 71:
            opt_flags = opt_flags + " -faligned-new"

    all_flags = common_flags + opt_flags + " -DNDEBUG"
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split()
    print(conf.env['CXXFLAGS'])

    if conf.options.exp:
        for i in conf.options.exp.split(','):
            Logs.pprint('NORMAL', 'configuring for exp: %s' % i)
            conf.recurse('exp/' + i)



def build(bld):
    bld.recurse('src/examples')

    install_files = []
    for root, dirnames, filenames in os.walk(bld.path.abspath()+'/src/'):
        if 'examples' in root:
            continue
        for filename in fnmatch.filter(filenames, '*.hpp'):
            install_files.append(os.path.join(root, filename))
    install_files = [f[len(bld.path.abspath())+1:] for f in install_files]

    for f in install_files:
        end_index = f.rfind('/')
        if end_index == -1:
            end_index = len(f)
        bld.install_files('${PREFIX}/include/' + f[4:end_index], f)

    if bld.options.exp:
        for i in bld.options.exp.split(','):
            Logs.pprint('NORMAL', 'Building exp: %s' % i)
            bld.recurse('exp/' + i)

