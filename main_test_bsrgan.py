import argparse
import csv
import logging
import os.path

import torch

from models.network_rrdbnet import RRDBNet as net
from utils import utils_image as util
from utils import utils_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BSRGAN', help='model name: BSRGAN, BSRGANx2, BSRNet')
    parser.add_argument('--input', type=str, default='testsets/RealSRSet', help='input LQ image directory')
    parser.add_argument('--gt', type=str, default=None, help='GT image directory for PSNR/SSIM evaluation')
    parser.add_argument('--output', type=str, default=None, help='output directory (default: <input>_results)')
    parser.add_argument('--csv', type=str, default=None, help='CSV output path (default: <output>/metrics.csv)')
    args = parser.parse_args()

    model_name = args.model
    sf = 2 if 'x2' in model_name.lower() else 4

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join('model_zoo', model_name + '.pth')
    logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    L_path = args.input
    E_path = args.output if args.output else args.input + f'_results_{model_name}'
    util.mkdir(E_path)

    logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    if args.gt:
        logger.info('{:>16s} : {:s}'.format('GT Path', args.gt))

    results = []

    for idx, img_path in enumerate(util.get_image_paths(L_path), 1):
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name + ext))

        img_L = util.imread_uint(img_path, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)

        util.imsave(img_E, os.path.join(E_path, img_name + '_' + model_name + '.png'))

        if args.gt:
            gt_path = os.path.join(args.gt, img_name + ext)
            if os.path.exists(gt_path):
                img_GT = util.imread_uint(gt_path, n_channels=3)
                psnr = util.calculate_psnr(img_E, img_GT)
                ssim = util.calculate_ssim(img_E, img_GT)
                logger.info('{:>16s} : {:.4f} dB / {:.4f}'.format('PSNR/SSIM', psnr, ssim))
                results.append({'filename': img_name + ext, 'psnr': psnr, 'ssim': ssim})
            else:
                logger.warning('GT not found: {:s}'.format(gt_path))

    if results:
        csv_path = args.csv if args.csv else os.path.join(E_path, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(results)

            avg_psnr = sum(r['psnr'] for r in results) / len(results)
            avg_ssim = sum(r['ssim'] for r in results) / len(results)
            writer.writerow({'filename': 'average', 'psnr': avg_psnr, 'ssim': avg_ssim})

        logger.info('Average PSNR: {:.4f} dB, SSIM: {:.4f}'.format(avg_psnr, avg_ssim))
        logger.info('CSV saved to: {:s}'.format(csv_path))


if __name__ == '__main__':
    main()
