import os
import sys

print('[Fooosti Main System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

from args_manager import args

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

import api_server  # noqa: E402  (torch-free HTTP bridge)
import uvicorn
uvicorn.run(api_server.app, host=args.listen, port=args.port, log_level='info')
