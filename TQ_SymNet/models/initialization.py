"""Fair quality-stage initialization, performed after runner setup and before training."""
import hashlib
import json
from pathlib import Path
from mmcv.runner import HOOKS, Hook
from TQ_SymNet.checkpoints import load_detector_checkpoint
from TQ_SymNet.data_contract import (load_split_contract, verify_checkpoint_contract,
                                    verify_dataset_contract)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@HOOKS.register_module()
class TQSplitContractHook(Hook):
    """Record cohort hashes in checkpoints and validate actual train/val loaders."""
    def __init__(self, manifest):
        self.manifest = str(manifest)
        self.contract = None

    def before_run(self, runner):
        self.contract, _ = load_split_contract(self.manifest)
        if runner.meta is None:
            runner.meta = {}
        previous = runner.meta.get('tq_data_contract')
        if previous is not None:
            verify_checkpoint_contract(previous, self.contract)
        runner.meta['tq_data_contract'] = self.contract
        runner.logger.info('[TQ-SPLIT] protocol=%s splitSeed=%d modelSeed=%s '
            'uniqueImages=479/120/200 trainSHA=%s valSHA=%s testSHA=%s',
            self.contract['protocol'], self.contract['split_seed'], runner.meta.get('seed'),
            self.contract['train_sha256'], self.contract['val_sha256'], self.contract['test_sha256'])

    def before_train_epoch(self, runner):
        verify_dataset_contract(runner.data_loader.dataset, self.contract['train_sha256'], 'train')
        for hook in runner.hooks:
            if hasattr(hook, 'dataloader'):
                verify_dataset_contract(hook.dataloader.dataset, self.contract['val_sha256'], 'validation')
        if runner.epoch == 0:
            runner.logger.info('[TQ-SPLIT-LOADERS] uniqueTrain=479 effectiveTrain=%d '
                'validation=120 reservedTest=200 repeatsAreTrainOnly=True',
                len(runner.data_loader.dataset))


@HOOKS.register_module()
class TQDetectorInitHook(Hook):
    """Use load_from=None; this hook deliberately ignores all scorer weights."""
    def __init__(self, checkpoint):
        self.checkpoint = str(checkpoint)

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if runner.epoch > 0 or runner.iter > 0:
            runner.logger.info('[TQ-INIT] resumed=True; existing detector/quality weights preserved')
            return
        if any(p.requires_grad for n, p in model.named_parameters()
               if not n.startswith('candidate_quality.')):
            raise RuntimeError('Matched quality ablations require a frozen detector')
        checkpoint, audit = load_detector_checkpoint(model, self.checkpoint)
        expected = runner.meta.get('tq_data_contract') if runner.meta else None
        if expected is None:
            raise RuntimeError('TQSplitContractHook must run before detector initialization')
        verify_checkpoint_contract(checkpoint.get('meta', {}).get('tq_data_contract'), expected)
        record = dict(detector_checkpoint=self.checkpoint,
                      detector_checkpoint_sha256=file_sha256(self.checkpoint),
                      quality_initialization='random_from_command_seed', **audit)
        runner.logger.info('[TQ-INIT] %s', json.dumps(record))
        if runner.meta is None:
            runner.meta = {}
        runner.meta['tq_initialization'] = record
        Path(runner.work_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(runner.work_dir) / 'initialization.json', 'w', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
