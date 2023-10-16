import os
import time

import paramiko


# def down_paramsens(sftp, remote_rt, overwrite=False):
#     exp_rt = remote_rt + '/training_data/param-sens'
#     def remote_file_iter():
#         for l1 in sftp.listdir(exp_rt):
#             for l2 in sftp.listdir(f'{exp_rt}/{l1}'):
#                 for l3 in sftp.listdir(f'{exp_rt}/{l1}/{l2}'):
#                     if l3 == 'gen_log':
#                         continue
#                     yield f'{exp_rt}/{l1}/{l2}/{l3}'
#     for path in remote_file_iter():
#         local_path = path.replace(f'{remote_rt}/training_data', local_rt).replace('/', '\\')
#         if os.path.exists(local_path) and not overwrite:
#             continue
#         local_fd, _ = os.path.split(local_path)
#         os.makedirs(local_fd, exist_ok=True)
#         sftp.get(path, local_path)
#         pass
#     # print([*remote_file_iter()])
#     # for
#     # for l, m, i in product(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5'), range(2, 6), range(1, 6)):
#     #     trial_path = exp_rt + f'/l{l}_m{m}/t{i}'
#     #     local_folder = f'{local_rt}\\param-sens\\l{l}_m{m}/t{i}'
#     #     print(local_folder)
#     #     os.makedirs(local_folder, exist_ok=True)
#     #     for item in param_sens_file_lists:
#     #         try:
#     #             sftp.get(f'{trial_path}/{item}', f'{local_folder}\\{item}')
#     #         except FileNotFoundError:
#     #             print(f'{trial_path}/{item}', 'not found')
#     #             pass
#     #     pass
#     pass

if __name__ == '__main__':
    ''' Servers '''
    host = '43.153.80.244'
    port = 22
    account = 'root'
    password = 'Mario001'
    remote_rt = '/root/MarioGit/MarioWeb/'
    local_rt = r'E:\academic\my_works\EDRL-TAC\EDRL-TAC-codes-v3\exp_data\survey data\formal\data'

    ''' Creat session '''
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, account, password, timeout=10)
    sftp = client.open_sftp()

    def down(folder, overwrite=False):
        for item in sftp.listdir(f'{remote_rt}/{folder}'):
            remote_item = f'{remote_rt}/{folder}/{item}'
            local_item = f'{local_rt}\\{folder}\\{item}'
            # yn = 'N'
            if not overwrite and os.path.exists(local_item):
                # yn = 'Y'
                continue
            # print(yn, remote_item)
            sftp.get(remote_item, local_item)
            time.sleep(0.1)
            # print(yn, item)
        pass

    down('jsons')
    down('reps')
    pass
