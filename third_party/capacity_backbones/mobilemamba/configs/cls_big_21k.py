from argparse import Namespace as _Namespace

# =========> dataset <=================================
data = _Namespace()
data.nb_classes = 10450

# =========> trainer <=================================
trainer = _Namespace()
trainer.epoch_full = 90
trainer.test_start_epoch = 80
trainer.test_per_epoch = 5
