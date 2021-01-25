import subprocess
import sys, json
import re
from psutil import Process
from signal import SIGTERM 


if __name__ == '__main__':
    ''' Code adapted from: 
    https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab
    
    For some reason, soemtimes you need to wait ~30 seconds and/or run this code twice for Tensorboard to successfully 
    be piped through to a public url. 
    '''
    
    port = 8008
    # Close tensorboard port if it's already hosting something else
    port_out = subprocess.Popen(f'sudo lsof -i :{port}', shell = True, stdout = subprocess.PIPE)
    port_out = [j.decode('utf-8') for j in port_out.stdout.readlines()][1::]
    pids_on_port = [int(i.split(' ')[1]) for i in port_out if len(i)>1]
    for pid in pids_on_port:
        p = Process(pid)
        p.terminate()

    LOG_DIR = '../../models/logs/20210122-204141'

    # Define commands to initialize tensorboard + ngrok, then set up a tunnerl
    cmd_str_tb = f'tensorboard --logdir {LOG_DIR} --host 0.0.0.0 --port {port} &'
    cmd_str_ngrok = f'~/ngrok http {port} &'
    cmd_str_pipe = 'curl -s http://localhost:4040/api/tunnels'

    # Run commands
    subprocess.Popen(cmd_str_tb, shell = True, stdout = subprocess.PIPE)
    subprocess.Popen(cmd_str_ngrok, shell = True, stdout = subprocess.PIPE)
    cmd_out = subprocess.Popen(cmd_str_pipe, shell = True, stdout = subprocess.PIPE)

    
    # Decode output
    # sub_dict = json.loads([j.decode('utf-8') for j in cmd_out.stdout.readlines()][0])
    sub_dict = [j.decode('utf-8') for j in cmd_out.stdout.readlines()][0]
    clean_out = json.loads(sub_dict)
    public_url = clean_out['tunnels'][0]['public_url']

    # Print public URL where tensorboard results are being hosted
    print(f'Public URL is at: {public_url}')
