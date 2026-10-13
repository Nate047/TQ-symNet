"""Linear 1cycle LR hook (Leslie Smith, arXiv:1708.07120).

Lives in mmdet.models.utils so `import mmdet.models.utils.one_cycle_hook`
always resolves (the mmdet package is installed).  Kept out of config files
because mmcv's _file2dict collects top-level classes as config entries and
yapf then fails to dump them.
"""

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module(force=True)
class OneCycleLrHook(Hook):
    """Linear 1cycle: lr_min -> lr_max over pct of iters, then back to lr_min.

    Runs at LOW priority (last) so it overrides any lr_config updater.
    """

    def __init__(self, lr_min=1e-5, lr_max=3e-3, pct=0.3):
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.pct = float(pct)
        self.priority = 'LOW'
        if not 0 < self.pct < 1:
            raise ValueError('pct must be in (0, 1)')

    def before_run(self, runner):
        self._total = runner.max_iters
        self._phase1 = int(self._total * self.pct)

    def before_train_iter(self, runner):
        it = runner.iter
        if it < self._phase1:
            frac = it / max(self._phase1 - 1, 1)
            lr = self.lr_min + (self.lr_max - self.lr_min) * frac
        else:
            frac = (it - self._phase1) / max(
                self._total - self._phase1 - 1, 1)
            lr = self.lr_max - (self.lr_max - self.lr_min) * frac
        for group in runner.optimizer.param_groups:
            group['lr'] = lr
