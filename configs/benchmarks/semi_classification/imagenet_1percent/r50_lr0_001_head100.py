_base_ = 'base.py'
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
                 paramwise_options={'\Ahead.': dict(lr_mult=100)})
