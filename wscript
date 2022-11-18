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
import eigen
import tbb
import avx


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('eigen')
    opt.load('tbb')

    opt.add_option('--no-native', action='store_true', help='Do not compile with march=native optimization flags', dest='no_native')

def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('eigen')
    conf.load('tbb')

    conf.check_eigen(required=True)
    conf.check_tbb(required=True)

    native = '-march=native'
    native_icc = 'mtune=native'

    if conf.options.no_native:
        native = ''
        native_icc = ''

    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -xHost " + native_icc + " -unroll -g"
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 " + native + " -g"
    else:
        gcc_version = int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1])
        if gcc_version < 47:
            conf.fatal('You need gcc version >= 4.7 for this project.')
        else:
            common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 " + native + " -g"

    all_flags = common_flags + opt_flags
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
    print(conf.env['CXXFLAGS'])

def build(bld):
    libs = 'EIGEN TBB '

    cxxflags = bld.get_env()['CXXFLAGS']


    bld.program(features = 'cxx',
                install_path = None,
                source = 'src/examples/sphere.cpp',
                includes = './src',
                uselib = libs,
                cxxflags = cxxflags,
                target = 'sphere')

    bld.program(features = 'cxx',
                install_path = None,
                source = 'src/examples/sphere_map_elites.cpp',
                includes = './src',
                uselib = libs,
                cxxflags = cxxflags,
                target = 'sphere_map_elites')

    bld.program(features = 'cxx',
                install_path = None,
                source = 'src/examples/sphere_de.cpp',
                includes = './src',
                uselib = libs,
                cxxflags = cxxflags,
                target = 'sphere_de')

    install_files = []
    for root, dirnames, filenames in os.walk(bld.path.abspath()+'/src/'):
        for filename in fnmatch.filter(filenames, '*.hpp'):
            install_files.append(os.path.join(root, filename))
    install_files = [f[len(bld.path.abspath())+1:] for f in install_files]

    for f in install_files:
        end_index = f.rfind('/')
        if end_index == -1:
            end_index = len(f)
        bld.install_files('${PREFIX}/include/' + f[4:end_index], f)
