import os
import cv2
import sys
import time
import math
import copy
import torch
import pickle
import shutil
import random
import logging
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from IPython import embed
from interfaces import base
from utils import utils_moran
from datetime import datetime
from utils.util import str_filt
from time import gmtime, strftime
from utils import util, ssim_psnr
from torchvision import transforms
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils.metrics import get_str_list, Accuracy
from torch.utils.tensorboard import SummaryWriter

to_pil = transforms.ToPILImage()


times = 0
easy_test_times = 0
medium_test_times = 0
hard_test_times = 0
class TextSR(base.TextBase):
    def train(self):
        pass

    def get_crnn_pred(self, outputs):
        alphabet = '-0123456789abcdefghijklmnopqrstuvwxyz'
        predict_result = []
        for output in outputs:
            max_index = torch.max(output, 1)[1]
            out_str = ""
            last = ""
            for i in max_index:
                if alphabet[i] != last:
                    if i != 0:
                        out_str += alphabet[i]
                        last = alphabet[i]
                    else:
                        last = ""
            predict_result.append(out_str)
        return predict_result


    def eval(self, model, val_loader, image_crit, index, recognizer, aster_info, mode):
        global easy_test_times
        global medium_test_times
        global hard_test_times

        for p in model.parameters():
            p.requires_grad = False
        for p in recognizer.parameters():
            p.requires_grad = False
        model.eval()
        recognizer.eval()
        n_correct = 0
        n_correct_lr = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0,
                       'images_and_labels': []}
        image_start_index = 0
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            images_lr = images_lr.to(self.device)
            images_hr = images_hr.to(self.device)
            images_sr = model(images_lr)

            if i == len(val_loader) - 1:
                index = random.randint(0, images_lr.shape[0]-1)
                self.writer.add_image(f'vis/{mode}/lr_image', images_lr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/sr_image', images_sr[index,...], easy_test_times)
                self.writer.add_image(f'vis/{mode}/hr_image', images_hr[index,...], easy_test_times)

            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

            recognizer_dict_sr = self.parse_crnn_data(images_sr[:, :3, :, :])
            recognizer_output_sr = recognizer(recognizer_dict_sr)
            outputs_sr = recognizer_output_sr.permute(1, 0, 2).contiguous()
            predict_result_sr = self.get_crnn_pred(outputs_sr)
            metric_dict['images_and_labels'].append(
                (images_hr.detach().cpu(), images_sr.detach().cpu(), label_strs, predict_result_sr))

            cnt = 0
            for pred, target in zip(predict_result_sr, label_strs):
                # self.logging.info('{} | {} | {} | {}\n'.format(write_line, pred, str_filt(target, 'lower'),
                #                                      pred == str_filt(target, 'lower')))
                # write_line += 1
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
                cnt += 1

            sum_images += val_batch_size
            torch.cuda.empty_cache()
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        # logging.info('[{}]\t'
        #       'loss_rec {:.3f}| loss_im {:.3f}\t'
        #       'PSNR {:.2f} | SSIM {:.4f}\t'
        #       .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #               0, 0,
        #               float(psnr_avg), float(ssim_avg), ))
        # logging.info('save display images')
        accuracy = round(n_correct / sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)
        # logging.info('sr_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg

        if mode == 'easy':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, easy_test_times)
            easy_test_times += 1
        if mode == 'medium':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, medium_test_times)
            medium_test_times += 1
        if mode == 'hard':
            self.writer.add_scalar('{}_accuracy'.format(mode), accuracy, hard_test_times)
            hard_test_times += 1

        return metric_dict

    def test(self):
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        items = os.listdir(self.test_data_dir)### items==['easy', 'hard', 'medium']

        my_acc={}

        for test_dir in items:
            test_data, test_loader = self.get_test_data(os.path.join(self.test_data_dir, test_dir))#加载测试数据

            data_name = self.args.test_data_dir.split('/')[-1]# test

            logging.info('evaling %s' % data_name)
            if self.args.rec == 'moran':
                moran = self.MORAN_init()
                moran.eval()
            elif self.args.rec == 'aster':
                aster, aster_info = self.Aster_init()
                aster.eval()
            elif self.args.rec == 'crnn':
                crnn, _ = self.CRNN_init()
                crnn.eval()
            if self.args.arch != 'bicubic':
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
            n_correct = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}#字典
            current_acc_dict = {test_dir: 0}
            time_begin = time.time()
            sr_time = 0
            for i, data in (enumerate(test_loader)):

                images_hr, images_lr, label_strs = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)#torch.Size([16, 3, 16, 64])
                images_hr = images_hr.to(self.device)#torch.Size([16, 3, 32, 128])
                sr_beigin = time.time()
                images_sr = model(images_lr)#torch.Size([16, 3, 32, 128])

                # #images_sr = torch.clamp(images_sr, 0, 1) 目前还不确定是否需要将范围限定在0到1
                # # print(111111111111111111111111,self.args.exp_name)
                #
                # path='checkpoint/'+self.args.exp_name+'/'+test_dir
                #
                # ###SR的保存
                # SR_left_img = transforms.ToPILImage()(torch.squeeze(images_sr.data.cpu(), 0))
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # SR_left_img.save(path + '/img%s.png'% i)#'evaling %s' % data_name
                #
                # ###LR的保存
                # lr = transforms.ToPILImage()(torch.squeeze(images_lr.data.cpu(), 0))
                # if not os.path.exists(path+'lr'):
                #     os.makedirs(path+'lr')
                # lr.save(path+'lr' + '/img%s.png'% i)#'evaling %s' % data_name
                #
                # #HR的保存
                # hr = transforms.ToPILImage()(torch.squeeze(images_hr.data.cpu(), 0))
                # if not os.path.exists(path+'hr'):
                #     os.makedirs(path+'hr')
                # hr.save(path+'hr' + '/img%s.png'% i)#'evaling %s' % data_name



                sr_end = time.time()
                sr_time += sr_end - sr_beigin
                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                if self.args.rec == 'moran':
                    moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                    moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                         debug=True)
                    preds, preds_reverse = moran_output[0]
                    _, preds = preds.max(1)
                    sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                    pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                elif self.args.rec == 'aster':

                    aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])

                    aster_output_sr = aster(aster_dict_sr)

                    pred_rec_sr = aster_output_sr['output']['pred_rec']
                    pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                    aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                    aster_output_lr = aster(aster_dict_lr)
                    pred_rec_lr = aster_output_lr['output']['pred_rec']
                    pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                elif self.args.rec == 'crnn':
                    crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                    crnn_output = crnn(crnn_input)
                    _, preds = crnn_output.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                    pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)#这个地方是识别后的字符
                for pred, target in zip(pred_str_sr, label_strs):

                    if str_filt(pred, 'lower') == str_filt(target, 'lower'):#判断字符是否相等
                        n_correct += 1

                sum_images += val_batch_size
                torch.cuda.empty_cache()
                if i % 10 == 0:
                    logging.info('Evaluation: [{}][{}/{}]\t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  i + 1, len(test_loader), ))
                # self.test_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, str_filt)
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])


            acc = round(n_correct / sum_images, 4)
            fps = sum_images / (time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[test_dir] = float(acc)
            my_acc[test_dir]=float(acc)

            result = {'accuracy': current_acc_dict, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}

            logging.info(result)
        print('average: ',(my_acc['easy'] * 1619 + my_acc['medium'] * 1411 + my_acc['hard'] * 1343) / 4373.0)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn, _ = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3],
                                        test=True,
                                        debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * 1)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * 1)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            logging.info('{} ===> {}'.format(pred_str_lr, pred_str_sr))
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        logging.info('fps={}'.format(fps))


if __name__ == '__main__':
    embed()
