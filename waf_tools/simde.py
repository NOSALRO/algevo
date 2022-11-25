#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2022

"""
Quick n dirty simde detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
    opt.add_option('--simde', type='string', help='path to Simde', dest='simde')

@conf
def check_simde(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]
    includes_check = ['/usr/include', '/usr/local/include', '/opt/include', '/opt/local/include', '/opt/proxsuite/include', '/sw/include', '/opt/homebrew/include']

    required = kw.get('required', False)

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.simde:
        includes_check = [conf.options.simde]

    try:
        conf.start_msg('Checking for Simde')
        incl = get_directory('simde/simde-arch.h', includes_check)
        conf.env.INCLUDES_SIMDE = [incl]
        conf.end_msg(incl)
    except:
        if required:
            conf.fatal('Not found in %s' % str(includes_check))
        conf.end_msg('Not found in %s' % str(includes_check), 'RED')
        return

    return
