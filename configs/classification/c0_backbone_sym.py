"""Official Sym convolutional classifier; detection backbone permanently frozen."""
backbone = dict(type='transnext_small', pretrain_size=384, img_size=512,
                pretrained=None, scales=5)
# M2 and the selected frozen A0 have identical detector backbones.
backbone_checkpoint = (
    '/home/data2/LiYingxian/TQ_SymNet/work_dirs/m2_s0/'
    'best_official_tb_bbox_mAP_epoch_13.pth')
input_channels = 576
data_root = '/home/data2/LiYingxian/Dataset_TB/TBX11K'
data = dict(
    train_annotation=data_root + '/annotations/json/TBX11K_train.json',
    val_annotation=data_root + '/annotations/json/TBX11K_val.json',
    test_annotation=data_root + '/annotations/json/all_test.json',
    image_root=data_root + '/imgs', image_size=512,
    mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375),
    flip_probability=.5, batch_size=8, val_batch_size=8, workers=4,
    expected_train_images=6600, expected_val_images=1800, expected_test_images=3302)
optimizer = dict(type='SGD', lr=.001, momentum=.9, weight_decay=.0001)
training = dict(epochs=12, lr_steps=(8, 11), lr_gamma=.1,
                warmup_iters=500, warmup_ratio=.001, class_weights=(1., 1., 4.))
selection_metric = 'tb_auc'
log_interval = 50
