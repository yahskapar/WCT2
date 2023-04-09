"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
from tqdm import tqdm
import argparse
import cv2
import numpy as np

import torch
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info, transform_ubfc_image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret

def get_seg(input_frame, model):
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_frame)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Generate the output predictions
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Create a color palette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((input_frame.shape[1], input_frame.shape[0]))
    r.putpalette(colors)

    return r

# Resize back to original size
def resize_to_original(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def face_detection(frame, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.
    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    detector = cv2.CascadeClassifier('/playpen-nas-ssd/akshay/UNC_Google_Physio/MA-rPPG-Video-Toolbox/utils/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        face_box_coor = np.argmax(face_zone, axis=0)
        face_box_coor = face_zone[face_box_coor[2]]
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    else:
        face_box_coor = face_zone[0]
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor

def read_ubfc_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        print(np.shape(frames))
        return np.asarray(frames)

def run_bulk(config, model):

    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')

    # # The filenames of the content and style pair should match
    # fnames = set(os.listdir(config.content)) & set(os.listdir(config.style))

    # if config.content_segment and config.style_segment:
    #     fnames &= set(os.listdir(config.content_segment))
    #     fnames &= set(os.listdir(config.style_segment))

    # for fname in tqdm.tqdm(fnames):
    #     if not is_image_file(fname):
    #         print('invalid file (is not image), ', fname)
    #         continue
    # _content = os.path.join(config.content, fname)
    # _style = os.path.join(config.style, fname)
    # _content_segment = os.path.join(config.content_segment, fname) if config.content_segment else None
    # _style_segment = os.path.join(config.style_segment, fname) if config.style_segment else None
    # _output = os.path.join(config.output, fname)

    # content = open_image(config.content, config.image_size).to(device)
    content = read_ubfc_video(config.content)

    cropped_frames = []
    face_region_all = []

    # First, compute the median bounding box across all frames
    for frame in content:
        face_box = face_detection(frame, True, 2.0) # MAUBFC and others
        face_region_all.append(face_box)
    face_region_all = np.asarray(face_region_all, dtype='int')
    face_region_median = np.median(face_region_all, axis=0).astype('int')

    # Apply the median bounding box for cropping and subsequent resizing
    for frame in content:
        cropped_frame = frame[int(face_region_median[1]):int(face_region_median[1]+face_region_median[3]),
                            int(face_region_median[0]):int(face_region_median[0]+face_region_median[2])]
        resized_frame = resize_to_original(cropped_frame, np.shape(content)[2], np.shape(content)[1])
        cropped_frames.append(resized_frame)

    content = cropped_frames
    # copy_of_np_content = content
    # print(np.shape(content))

    restyled_video = []    
    frame_count = np.shape(content)[0]
    frames_pbar = tqdm(list(range(frame_count)))

    for frame in content:
        content_frame = transform_ubfc_image(Image.fromarray(frame)).to(device)
        style = open_image(config.style, config.image_size).to(device)
        # content_seg = get_seg(cv2.cvtColor(cv2.imread(config.content), cv2.COLOR_BGR2RGB), model)
        content_seg = get_seg(frame, model)
        style_seg = get_seg(cv2.cvtColor(cv2.imread(config.style), cv2.COLOR_BGR2RGB), model)
        content_segment = load_segment(content_seg, config.image_size)
        style_segment = load_segment(style_seg, config.image_size)     
        # _, ext = os.path.splitext(config.content)
        # _output = os.path.join(config.output, "Test.png")

        save_image(content_frame.clamp_(0, 1), os.path.join(config.output, "content_frame_debug.png"), padding=0)
        save_image(style.clamp_(0, 1), os.path.join(config.output, "style_debug.png"), padding=0)
        
        if not config.transfer_all:
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                # postfix = '_'.join(sorted(list(transfer_at)))
                # fname_output = _output.replace(ext, '_{}_{}{}'.format(config.option_unpool, postfix, ext))
                # print('------ transfer:', _output)
                wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
                with torch.no_grad():
                    img = wct2.transfer(content_frame, style, content_segment, style_segment, alpha=config.alpha)
                # save_image(img.clamp_(0, 1), fname_output, padding=0)
        img = F.interpolate(img, size=(480, 640), mode='area')
        img = img.clamp_(0, 1)
        # save_image(img, os.path.join(config.output, "output_debug_resized.png"), padding=0)
        img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        restyled_video.append(img)
        frames_pbar.update(1)
    frames_pbar.close()
    save_path = os.path.join(config.output, "Test.npy")
    np.save(save_path, restyled_video)

    # Write the video as an mp4 file
    out_path = os.path.join(config.output, "Test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = restyled_video[0].shape
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
    for frame in restyled_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    cv2.imwrite(os.path.join(config.output, "Test_debug.png"), cv2.cvtColor(restyled_video[777], cv2.COLOR_RGB2BGR))
    # else:
    #     for _transfer_at in get_all_transfer():
    #         with Timer('Elapsed time in whole WCT: {}', config.verbose):
    #             postfix = '_'.join(sorted(list(_transfer_at)))
    #             fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
    #             print('------ transfer:', fname)
    #             wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
    #             with torch.no_grad():
    #                 img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
    #             save_image(img.clamp_(0, 1), fname_output, padding=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./examples/content')
    parser.add_argument('--content_segment', type=str, default=None)
    parser.add_argument('--style', type=str, default='./examples/style')
    parser.add_argument('--style_segment', type=str, default=None)
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()

    print(config)

    # Load the segmentation model on GPU device 2
    device = torch.device('cuda:2')
    model = torch.hub.load('pytorch/vision:v0.9.2', 'deeplabv3_resnet50', pretrained=True)
    model.to(device)
    model.eval()

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    run_bulk(config, model)