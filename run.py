import sys
import runpy

import warnings
warnings.filterwarnings("ignore")



for i in range(3):  # 独立run 3 次

    args = """python -m laruGL.MODEL.train
             --setting transductive
             --split_mode dfs
             --size_subg_edge 1000
             --device cuda:1"""
      
    args = args.split()

    if args[0] == 'python':
        """pop up the first in the args"""
        args.pop(0)

    if args[0] == '-m':
        """pop up the first in the args"""
        args.pop(0)

        fun = runpy.run_module

    else:
        fun = runpy.run_path

    sys.argv.extend(args[1:])

    fun(args[0], run_name='__main__')

    

