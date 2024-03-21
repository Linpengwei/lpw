import torch
import torch.nn as nn
import numpy as np
import yaml
import cv2
from torchvision.transforms import Resize
import urllib.request as request
import base64
import time
import sys
sys.path.append('./')

from utils.extern.yolov7_extern import letterbox, non_max_suppression, scale_coords
from utils.box_transform import box_trans,xyxy2xywh

from model.DecoderFactory import Decoder_map
from thop import profile, clever_format
from peft import LoraConfig, get_peft_model

class PipelineInference_pose_check:
    def __init__(self, det_cfg="", base_cfg="", decoder_cfg=""):
        super().__init__()
        
        self.cfg_det         = None
        self.cfg_base        = None
        
        self.det_model        = None
        self.recg_pre_process = {}
        self.base_model       = {}
        self.decoder_models   = {}



        #load detection jit
        with open(det_cfg,'r') as f:
            self.cfg_det = yaml.load(f,Loader=yaml.FullLoader)
        self.det_model  = torch.jit.load(self.cfg_det['path']).cuda().half()
        self.det_model.eval()   # <class 'torch.jit._script.RecursiveScriptModule'>

            
        #load base jit
        with open(base_cfg,'r') as f:
            self.cfg_base = yaml.load(f,Loader=yaml.FullLoader)
        for model_name in self.cfg_base:
            model_info = self.cfg_base[model_name]
            self.recg_pre_process[model_name]={}
            self.recg_pre_process[model_name]['resize_fn']  = Resize(model_info['input_shape'],antialias=True)
            self.recg_pre_process[model_name]['input_type'] = model_info['input_type']
            self.base_model[model_name] = torch.jit.load(model_info['path']).cuda().half()
        
        #load decoders
        if isinstance(decoder_cfg,str):
            with open(decoder_cfg,'r') as f:
                self.decoder_cfg = yaml.load(f,Loader=yaml.FullLoader)
                for func_name in self.decoder_cfg:
                    model_path = self.decoder_cfg[func_name]['model_path']
                    self.decoder_models[func_name] = torch.jit.load(model_path).cuda().half()
        else:
                for func_name in decoder_cfg:
                    cfg_file, model_path = decoder_cfg[func_name]

                    with open(cfg_file,'r') as f:
                        cfg_decoder = yaml.load(f,Loader=yaml.FullLoader)

                    model_key   = cfg_decoder['model']['decoder_key']
                    decoder_fn  = Decoder_map[model_key]
                    act_decoder = decoder_fn(cfg_decoder)
                    model_params = torch.load(model_path)   # torch data
                    act_decoder.load_state_dict(model_params)  # network
                    # print(act_decoder.training)
                    act_decoder = act_decoder.half().cuda().eval()
                    # print(act_decoder.training)
                    
                    
                    self.decoder_models[func_name] = act_decoder

    def inference_path(self,image_path, res_norm=False, res_xywh=False):
        image_rgb = cv2.imread(image_path)[:,:,::-1]
        return self.forward(image_rgb, res_norm, res_xywh)

    def inference_url(self,image_url, res_norm=False, res_xywh=False):
        try:
            response = request.urlopen(image_url,timeout=2)
        except:
            return None
        img_array = np.array(bytearray(response.read()), dtype=np.uint8)
        image_rgb = cv2.imdecode(img_array, -1)[:,:,::-1]
        if image_rgb is None:
            return None
        
        return self.forward(image_rgb, res_norm, res_xywh)
    
    def inference_b64(self,b64_str, res_norm=False, res_xywh=False):
        b64_str_decode = base64.b64decode(b64_str)
        img_array = np.frombuffer(b64_str_decode,np.uint8)
        image_rgb = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)[:,:,::-1]
        if image_rgb is None:
            return None
        
        return self.forward(image_rgb, res_norm, res_xywh)


    def inference_b64_v1(self,b64_str, res_norm=False, res_xywh=False):
        base64_bytes = bytes(b64_str, encoding="utf8")
        imgString = base64.b64decode(base64_bytes)
        image = self.jpeg.decode(imgString)

        return self.forward(image, res_norm, res_xywh)


    def pos_check(self, pos_feature:torch.Tensor):

        batch_size, key_point_num, _, _= pos_feature.shape
        reshape_pose_feature = pos_feature.reshape([batch_size,key_point_num,-1])
        keypoint_prob,_ = torch.max(reshape_pose_feature,dim=2)
        # print("keypoint_prob is : ", keypoint_prob)
        #check phonecall 
        left_ear_check  = keypoint_prob[:,3] > 0.7
        right_ear_check = keypoint_prob[:,4] > 0.7

        left_shoulder_check = keypoint_prob[:,5] > 0.7
        right_shoulder_check = keypoint_prob[:,6] > 0.7

        left_elbow_check = keypoint_prob[:,7] > 0.7
        right_elbow_check = keypoint_prob[:,8] > 0.7

        left_knee_check = keypoint_prob[:,13] > 0.7
        right_knee_check = keypoint_prob[:,14] > 0.7
        # print("keypoints prob:", keypoint_prob)
        
        # check fall
        fall_flag = torch.logical_and(left_knee_check, right_knee_check).detach().cpu().numpy()
        # print("fall_flag输出：", fall_flag)

        return {'fall': fall_flag}
            
    def forward(self, image_rgb, res_norm=False, res_xywh=False):
        
        start_time = time.time()
        with torch.no_grad():
            
            #inference detection model
            boxes_np      = self.inference_yolov7(image_rgb)
            boxes_num     = boxes_np.shape[0]
            if boxes_num == 0:
                return [{}, boxes_np, {}]
            # print('detection time cost:', time.time() - start_time)
            #inference basemodel
            mid_datas  = self.prep_box_img(image_rgb,boxes_np)
            decoder_input = {}
            for model_name in self.base_model:
                decoder_input[model_name] = self.base_model[model_name](mid_datas[model_name])


                # model1 = torch.jit.trace(self.base_model['pos'], mid_datas['pos'])

            #inference decoders
            cls_result = {}
            for func_name in self.decoder_models:
                cls_result[func_name] = self.decoder_models[func_name](decoder_input['vit'], decoder_input['pos'])[1]

                # Inference: flops2: 85.40M, params2: 4.74M
                # flops2, params2 = profile(self.decoder_models[func_name], inputs=(decoder_input['vit'], decoder_input['pos']))
                # print('Inference: flops2: {:.2f}M, params2: {:.2f}M'.format(flops2 / 1e6, params2 / 1e6))



        #output coordinate adjust
        torch.cuda.empty_cache()
        
        
          
        if res_norm:
            h,w,_ = image_rgb.shape
            boxes_np[:,0] = boxes_np[:,0] / w
            boxes_np[:,1] = boxes_np[:,1] / h
            boxes_np[:,2] = boxes_np[:,2] / w
            boxes_np[:,3] = boxes_np[:,3] / h
        if res_xywh:
            boxes_np = xyxy2xywh(boxes_np)
            
        pose_check_result = self.pos_check(decoder_input['pos'])

        print('all time cost:', time.time() - start_time)

        return cls_result, boxes_np, pose_check_result
    
    def inference_yolov7(self, image_rgb):
        resized_image,resize_ratio,pad_size = letterbox(image_rgb,640,auto=False,stride=32)
        resized_image    = resized_image / 255.
        det_input_tensor = torch.from_numpy(resized_image).half().cuda().permute(2,0,1).unsqueeze(0)
        
        model_pred       = self.det_model(det_input_tensor)[0].detach()

        # flops1, params1 = profile(self.det_model, det_input_tensor)
        # print('Inference: flops1: {:.2f}M, params1: {:.2f}M'.format(flops1 / 1e6, params1 / 1e6))

        box_preds        = non_max_suppression(model_pred,classes=self.cfg_det['classes'])

        for box_pred in box_preds:
            det = scale_coords(det_input_tensor.shape[2:], box_pred[:,:4], image_rgb.shape).round()
        boxes_np   = box_preds[0].detach().cpu().numpy()  #size = (num_box, 6) x1,y1,x2,y2,conf_score,class_label
        
        boxes_np_list = []
        box_num = boxes_np.shape[0]
        image_height = image_rgb.shape[0]
        box_height = boxes_np[:,3] - boxes_np[:,1]
        for i in range(box_num):
            if box_height[i] > image_height * self.cfg_det['min_box_height']:
                boxes_np_list.append(boxes_np[i,:])
                
        if len(boxes_np_list) > 0:
            return np.stack(boxes_np_list)
        else:
            return np.zeros([0,6],dtype=np.float32)

    def prep_box_img(self, image_rgb, boxes_np):
        
        box_num = boxes_np.shape[0]
        input_img_map = {}
        img_h,img_w,_ = image_rgb.shape
        
        for i in range(box_num):
            box_list = boxes_np[i,:4].tolist()
            box_list = [int(coord) for coord in box_list]
            
            x1,y1,x2,y2 = box_trans(box_list,img_w,img_h,[0.5,0.3],norm=False)
            box_image = image_rgb[y1:y2,x1:x2,:].copy()
        
            for model_name in self.recg_pre_process:
                model_info = self.recg_pre_process[model_name]
                resize_fn = model_info['resize_fn']
                if model_info['input_type'] == 'MAE':
                    img_tensor = torch.from_numpy(box_image).cuda()
                    img_tensor = img_tensor.permute(2, 0, 1) / 255.
                    img_tensor = resize_fn(img_tensor.half()).unsqueeze(0)
                
                if model_name not in input_img_map:
                    input_img_map[model_name] = []
                input_img_map[model_name].append(img_tensor)
                
        output_map = {}
        for model_name in input_img_map:
            output_map[model_name] = torch.cat(input_img_map[model_name],axis=0)
            
        return output_map
    

        
if __name__ == '__main__':
    

    # pipe = PipelineInference_pose_check(
    #     './config/det.yaml',
    #     '../../../Services/test_pipeline_model/base_models.yaml',
    #     '../../../Services/test_pipeline_model/decoders.yaml'
    #     )
    with open('./config/Service_oldversion.yaml', 'r') as f:
        service_cfg = yaml.load(f, Loader=yaml.FullLoader)

    pipe = PipelineInference_pose_check(det_cfg=service_cfg['det_cfg'], base_cfg=service_cfg['base_cfg'],
                             decoder_cfg=service_cfg['decoder_cfg'])

    test_image = '/data1/pengwei/DetectFallDatasets/海南误报数据/20240112hn摔倒视频/screenshot/frame_2200.png'
    
    cls_result = pipe.inference_path(test_image)
    print(cls_result)
    