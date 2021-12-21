import torch
import argparse
import mmcv
from mmcv import DictAction
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils.models import modelFromConfig

def parse_args():
    parser = argparse.ArgumentParser("Agurment of convert_to_mobile.py")
    parser.add_argument('--cfg-path', required=True)
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    parser.add_argument('--device', default='cuda')

    parser.add_argument('--checkpoint-path', default='')
    
    parser.add_argument('--save-model', default='model.ptl')
    return parser.parse_args()
    

def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.cfg_path)
    config.merge_from_dict(args.cfg_options)

    model = modelFromConfig(cfg=config, checkpoint=args.checkpoint_path, device=args.device)
    model.eval()
    example_input = torch.rand(1,1, 17, 48, 56, 56).to(args.device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(args.save_model)


if __name__ == "__main__":
    main()