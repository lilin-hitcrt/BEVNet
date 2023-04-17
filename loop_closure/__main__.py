# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m loop_closure方式直接执行。

Authors: dingwendong(dingwendong@baidu.com)
Date:    2022/06/20 09:36:19
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from loop_closure.cmdline import main
sys.exit(main())
