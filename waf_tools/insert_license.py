#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2024 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2023-2024 Laboratory of Automation and Robotics, University of Patras, Greece
#|    Copyright (c) 2022-2024 Konstantinos Chatzilygeroudis
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
import fnmatch,re
import os, shutil, sys

license = '''Copyright (c) 2022-2024 Computational Intelligence Lab, University of Patras, Greece
Copyright (c) 2023-2024 Laboratory of Automation and Robotics, University of Patras, Greece
Copyright (c) 2022-2024 Konstantinos Chatzilygeroudis
Authors:  Konstantinos Chatzilygeroudis
email:    costashatz@gmail.com
website:  https://nosalro.github.io/
          http://cilab.math.upatras.gr/

This file is part of algevo.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

def make_dirlist(folder, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        if 'build' in root or '.waf3' in root:Computational Intelligence Lab, University of Patras, Greece
            continue
        for ext in extensions:
            for filename in fnmatch.filter(filenames, '*' + ext):
                matches.append(os.path.join(root, filename))
    return matches

def insert_header(fname, prefix, postfix, license, kept_header = []):
    input = open(fname, 'r')
    ofname = '/tmp/' + fname.split('/')[-1]
    output = open(ofname, 'w')
    for line in kept_header:
        output.write(line + '\n')
    output.write(prefix + '\n')
    has_postfix = len(postfix)>0
    my_prefix = prefix
    if has_postfix:
        my_prefix = ''
    for line in license.split('\n'):
        if len(line)>0:
            output.write(my_prefix + '    ' + line + '\n')
        else:
            output.write(my_prefix + '\n')
    if has_postfix:
        output.write(postfix + '\n')
    in_header = False
    for line in input:
        header = len(list(filter(lambda x: x == line[0:len(x)], kept_header))) != 0
        check_prefix = (line[0:len(prefix)] == prefix)
        check_postfix = (has_postfix and (line[0:len(postfix)] == postfix))
        if check_prefix and has_postfix:
            in_header = True
        if check_postfix:
            in_header = False
        if (not in_header) and (not check_prefix) and (not header) and (not check_postfix):
            output.write(line)
    output.close()
    shutil.move(ofname, fname)

def insert(directory):
    # cpp
    cpp =  make_dirlist(directory, ['.hpp', '.cpp', '.h', '.c', '.cc'])
    for i in cpp:
        insert_header(i, '//|', '', license)
    # py
    py = make_dirlist(directory, ['.py'])
    for i in py:
        insert_header(i, '#|', '', license, ['#!/usr/bin/env python', '# encoding: utf-8'])
    # CMake
    cmake = make_dirlist(directory, ['CMakeLists.txt'])
    for i in cmake:
        # metapackages should not have any comments
        if i.endswith('iiwa_ros/CMakeLists.txt'):
            continue
        insert_header(i, '#|', '', license)
    # # XML/URDF
    xml_urdf = make_dirlist(directory, ['.xml', '.urdf', '.xacro', '.launch'])
    for i in xml_urdf:
        header = ['<?xml version="1.0"?>']
        insert_header(i, '<!--|', '|-->', license, header)

if __name__ == '__main__':
    insert('.')