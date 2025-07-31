import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mscrobotics2425laptop11/ws_dis/install/rppg_node'
