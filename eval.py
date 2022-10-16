
from basicsr.utils.img_util import tensor2img
import yaml
from basicsr.models.archs.deformable_arch import deformable
from basicsr.data.paired_image_dataset import Dataset_PairedImage
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision import transforms
from PIL import Image
import os

def eval_model(ckp_path, config_path, visualize=False):
    print("load ckp: " + str(ckp_path))
    ckp_name = ckp_path.split("/")[-1][0:-4]
    # cfg = torch.load(str(ckp_path), map_location='cpu')
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if visualize:
        if not os.path.exists('./visualization/' + ckp_name):
            os.mkdir('./visualization/' + ckp_name)
        print("start visualization.")

    model = deformable(inp_channels= cfg['network_g']['inp_channels'], 
                        out_channels=cfg['network_g']['out_channels'], 
                        dim = cfg['network_g']['dim'],
                        num_blocks = cfg['network_g']['num_blocks'], 
                        num_refinement_blocks = cfg['network_g']['num_refinement_blocks'],
                        heads = cfg['network_g']['heads'],
                        groups = cfg['network_g']['groups'],
                        ffn_expansion_factor = cfg['network_g']['ffn_expansion_factor'],
                        bias = cfg['network_g']['bias'],
                        LayerNorm_type = cfg['network_g']['LayerNorm_type'],   ## Other option 'BiasFree'
                        dual_pixel_task = cfg['network_g']['dual_pixel_task'] )

    model.load_state_dict(torch.load(ckp_path, map_location='cpu')['params'])
    print("load checkpoint" + str(ckp_path))
    
    model = model.to(device)
    eval_dataset = Dataset_PairedImage(cfg['datasets']['val'])
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size = 1,
                                shuffle = False)
    
    model.eval()

    with torch.no_grad():
        loss_test_total = []
        psnr_y_total = []
        ssim_y_total = []
        psnr_total = []
        ssim_total = []
        data_length = len(eval_dataset)
        for test_data in tqdm(eval_dataloader):
            lq_test = test_data['lq'].to(device)
            gt_test = test_data['gt'].to(device)
            output_test = model(lq_test)
            # loss_test = loss_fn(gt_test, output_test)
            # loss_test_total.append(loss_test.cpu().item())

            psnr_y = calculate_psnr(gt_test, output_test, crop_border=0, input_order='CHW', test_y_channel=True)
            # ssim_y = calculate_ssim(gt_test*255, output_test*255, crop_border=0, input_order='CHW', test_y_channel=True)
            psnr_y_total.append(psnr_y)
            # ssim_y_total.append(ssim_y)

            # psnr = calculate_psnr(gt_test, output_test, crop_border=0, input_order='CHW', test_y_channel=False)
            # ssim = calculate_ssim(gt_test, output_test, crop_border=0, input_order='CHW', test_y_channel=False)
            # psnr_total.append(psnr)
            # ssim_total.append(ssim)
            if visualize:
                lq_path = test_data['lq_path'][0]
                visualizion(lq_test, output_test ,gt_test, ckp_name, lq_path)

        # loss_mean = sum(loss_test_total)/data_length
        psnr_y_mean = sum(psnr_y_total)/data_length
        # ssim_y_mean = sum(ssim_y_total)/data_length
        # psnr_mean = sum(psnr_total)/data_length
        # ssim_mean = sum(ssim_total)/data_length

        # print("test_psnr: "+str(psnr_mean))
        print("test_psnr_y: "+str(psnr_y_mean))
        # print("test_psnr on y channel: "+str(psnr_y_mean)+"  best_ssim: "+str(ssim_y_mean))


def visualizion(input, output ,gt, ckp_name, lq_path):
    tensor2img = transforms.ToPILImage()

    input = tensor2img(input.squeeze(0).cpu().clone())
    output = tensor2img(output.squeeze(0).cpu().clone())
    gt = tensor2img(gt.squeeze(0).cpu().clone())

    size = input.size[0]
    print(size)
    joint = Image.new('RGB', (size*3, size))
    loc1, loc2, loc3 = (0, 0), (size, 0), (size*2, 0)
    joint.paste(input, loc1)
    joint.paste(output, loc2)
    joint.paste(gt, loc3)

    joint.save("".join(["./visualization/", str(ckp_name), "/", str(lq_path), ".png"]))



if __name__ == '__main__':
    ckp_path = "./experiments/001_RealDenoising_DeformableRestormer_debug/models/net_g_16.pth"
    config_path = "./Denoising/Options/RealDenoising_Restormer.yml"
    eval_model(ckp_path, config_path, visualize=False)

    
        