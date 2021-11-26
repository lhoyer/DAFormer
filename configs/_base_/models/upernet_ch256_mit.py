_base_ = ['upernet_mit.py']

model = dict(decode_head=dict(channels=256, ))
