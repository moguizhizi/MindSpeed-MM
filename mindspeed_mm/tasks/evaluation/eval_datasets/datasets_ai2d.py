from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import BaseEvalDataset


class AI2DEvalDataset(BaseEvalDataset):

    def __init__(self, dataset_path, dataset_name):
        super().__init__(dataset_path, dataset_name)