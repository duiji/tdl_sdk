#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iterator>

#include <core/core/cvtdl_errno.h>
#include <error_msg.hpp>
#include <iostream>
#include "coco_utils.hpp"
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "object_utils.hpp"
#include "yolov8.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "image_utils.hpp"
//#include <zbar.h>
#include <set>
#include <string.h>

namespace cvitdl {
static void convert_det_struct(const Detections &dets, cvtdl_object_t *obj, int im_height,
                               int im_width) {
  CVI_TDL_MemAllocInit(dets.size(), obj);
  obj->height = im_height;
  obj->width = im_width;
  memset(obj->info, 0, sizeof(cvtdl_object_info_t) * obj->size);

  for (uint32_t i = 0; i < obj->size; ++i) {
    obj->info[i].bbox.x1 = dets[i]->x1;
    obj->info[i].bbox.y1 = dets[i]->y1;
    obj->info[i].bbox.x2 = dets[i]->x2;
    obj->info[i].bbox.y2 = dets[i]->y2;
    obj->info[i].bbox.score = dets[i]->score;
    obj->info[i].classes = dets[i]->label;
  }
}

template <typename T>
inline void parse_cls_info(T *p_cls_ptr, int num_anchor, int num_cls, int anchor_idx,
                           int cls_offset, float qscale, float *p_max_logit, int *p_max_cls) {
  int max_logit_c = -1;
  float max_logit = -1000;
  for (int c = 0; c < num_cls; c++) {
    float logit = p_cls_ptr[(c + cls_offset) * num_anchor + anchor_idx];
    if (logit > max_logit) {
      max_logit = logit;
      max_logit_c = c;
    }
  }
  *p_max_logit = max_logit * qscale;
  *p_max_cls = max_logit_c;
}

YoloV8Detection::YoloV8Detection() : YoloV8Detection(std::make_pair(64, 80)) {}

YoloV8Detection::YoloV8Detection(PAIR_INT yolov8_pair) {
  for (int i = 0; i < 3; i++) {
    m_preprocess_param[0].factor[i] = 0.003922;
    m_preprocess_param[0].mean[i] = 0.0;
  }
  m_preprocess_param[0].format = PIXEL_FORMAT_RGB_888_PLANAR;

  m_box_channel_ = yolov8_pair.first;
  alg_param_.cls = yolov8_pair.second;
}

// would parse 3 cases,1:box,cls seperate feature map,2 box+cls seperate featuremap,3 output decoded
// results
int YoloV8Detection::onModelOpened() {
  CVI_SHAPE input_shape = getInputShape(0);
  int input_h = input_shape.dim[2];
  int input_w = input_shape.dim[3];
  strides.clear();
  size_t num_output = getNumOutputTensor();
  for (size_t j = 0; j < num_output; j++) {
    CVI_SHAPE oj = getOutputShape(j);
    TensorInfo oinfo = getOutputTensorInfo(j);
    int feat_h = oj.dim[2];
    int feat_w = oj.dim[3];
    int channel = oj.dim[1];
    int stride_h = input_h / feat_h;
    int stride_w = input_w / feat_w;

    if (stride_h == 0 && num_output == 2) {
      if (channel == alg_param_.cls) {
        class_out_names[stride_h] = oinfo.tensor_name;
        strides.push_back(stride_h);
        LOGI("parse class decode branch:%s,channel:%d\n", oinfo.tensor_name.c_str(), channel);
      } else {
        bbox_out_names[stride_h] = oinfo.tensor_name;
        LOGI("parse box decode branch:%s,channel:%d\n", oinfo.tensor_name.c_str(), channel);
      }
      continue;
    }

    if (stride_h != stride_w) {
      LOGE("stride not equal,stridew:%d,strideh:%d,featw:%d,feath:%d\n", stride_w, stride_h, feat_w,
           feat_h);
      return CVI_FAILURE;
    }
    if (channel == m_box_channel_) {
      bbox_out_names[stride_h] = oinfo.tensor_name;
      strides.push_back(stride_h);
      LOGI("parse box branch,name:%s,stride:%d\n", oinfo.tensor_name.c_str(), stride_h);
    } else if (alg_param_.cls == 0 && num_output == 6) {
      alg_param_.cls = channel;
      class_out_names[stride_h] = oinfo.tensor_name;
      LOGI("parse class branch,name:%s,stride:%d,num_cls:%d\n", oinfo.tensor_name.c_str(), stride_h,
           channel);
    } else if (channel == alg_param_.cls) {
      class_out_names[stride_h] = oinfo.tensor_name;
      LOGI("parse class branch,name:%s,stride:%d\n", oinfo.tensor_name.c_str(), stride_h);
    } else if (channel == (m_box_channel_ + alg_param_.cls)) {
      strides.push_back(stride_h);
      bbox_class_out_names[stride_h] = oinfo.tensor_name;
      LOGI("parse box+class branch,name: %s,stride:%d\n", oinfo.tensor_name.c_str(), stride_h);
    } else {
      LOGE("unexpected branch:%s,channel:%d\n", oinfo.tensor_name.c_str(), channel);
      return CVI_FAILURE;
    }
  }

  return CVI_TDL_SUCCESS;
}

YoloV8Detection::~YoloV8Detection() {}

int YoloV8Detection::inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *obj_meta) {
  std::vector<VIDEO_FRAME_INFO_S *> frames = {srcFrame};
  int ret = run(frames);
  if (ret != CVI_TDL_SUCCESS) {
    LOGW("YoloV8Detection run inference failed\n");
    return ret;
  }
  CVI_SHAPE shape = getInputShape(0);
  if (strides.size() == 3) {
    outputParser(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
                 srcFrame->stVFrame.u32Height, obj_meta);
  } else {
    parseDecodeBranch(shape.dim[3], shape.dim[2], srcFrame->stVFrame.u32Width,
                      srcFrame->stVFrame.u32Height, obj_meta);
  }

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}

/**
 * 滑动窗检测
 */
int YoloV8Detection::windows_inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *obj_meta) {
  CVI_SHAPE shape = getInputShape(0);
  //////////////////////////////////
  //将一帧切割
  CVI_S32 s32Ret = CVI_SUCCESS;
  //1.初始
  //算法训练的长宽 640 * 640
  const int  algo_width = shape.dim[3]; 
  const int  algo_height = shape.dim[2];
  //传入帧长宽 1920 * 1080
  //暂时只有200万像素 后续会增加到 800万
  const int  frame_width = srcFrame->stVFrame.u32Width;
  const int  frame_height = srcFrame->stVFrame.u32Height; 

  //水平 百分之30重叠
  const int x_percent = 30;
  //垂直 百分之0重叠
  const int y_percent = 0;
  //水平方向步长
  int stride_x = algo_width * (100 - x_percent) / 100;
  //垂直方向步长
  int stride_y = algo_height * (100 - y_percent) / 100;

  Detections vec_obj;
  int run_ai_count = 0;
  int all_det_num = 0;
  for (int y = 0; y < frame_height; y += stride_y) {
    if (y + algo_height > frame_height) {
      y = frame_height - algo_height;
    }    

    for (int x = 0; x < frame_width; x += stride_x) {
      if (x + algo_width > frame_width) {
        x = frame_width - algo_width;
      }

      VIDEO_FRAME_INFO_S subFrame;

      VPSS_CROP_INFO_S cropAttr;
      cropAttr.bEnable = true;
      cropAttr.stCropRect = {x, y, algo_width, algo_height};
      VPSS_CHN_ATTR_S chnAttr;
      VPSS_CHN_DEFAULT_HELPER(&chnAttr, algo_width, algo_height, PIXEL_FORMAT_RGB_888, false);
      s32Ret = mp_vpss_inst->sendCropChnFrame(srcFrame, &cropAttr, &chnAttr, 1);
      
      s32Ret = mp_vpss_inst->getFrame(&subFrame, 0, 2000);

      std::vector<VIDEO_FRAME_INFO_S *> frames = {&subFrame};
      int ret = run(frames);
      if (ret != CVI_TDL_SUCCESS) {
        return ret;
      }

      int det_num = 0;

      det_num = outputParser_windowsDetect(frame_width, frame_height, &vec_obj);

      // 对于检测到的做坐标转换,转换到原图
      for (int j = 0; j < det_num; j++) {
          PtrDectRect &det = vec_obj[all_det_num + j];
          det->x1 += x;
          det->y1 += y;
          det->x2 += x;
          det->y2 += y;
          //printf("x1=%f y1=%f x2=%f y2=%f score=%f \n", det->x1, det->y1, det->x2, det->y2, det->score);
      }

      all_det_num += det_num;

      // 释放帧
      CVI_VPSS_ReleaseChnFrame(0, 0, &subFrame);

      run_ai_count ++;
      if (x + algo_width >= frame_width){
        break;
      }
    }

    if (y + algo_height >= frame_height){
      break;
    }
  }

  postProcess(vec_obj, frame_width, frame_height, obj_meta, false);

  model_timer_.TicToc("post");
  return CVI_TDL_SUCCESS;
}

/*
// zbar接口
std::set<std::string> decode(const cv::Mat image) {
    // 创建 ZBar 图像扫描器
    zbar::ImageScanner scanner;

    scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);

    // 将 OpenCV 图像转换为 ZBar 图像
    zbar::Image zbarImage(image.cols, image.rows, "Y800", image.data, image.cols * image.rows);

    // 扫描条形码和二维码
    int n = scanner.scan(zbarImage);

    // 解析扫描结果
    std::set<std::string> decodedTextSet;
    if (n > 0) {
        for (auto symbol = zbarImage.symbol_begin(); symbol != zbarImage.symbol_end(); ++symbol) {
            // 获取解码结果
            decodedTextSet.insert(symbol->get_data());
        }
    }
    return decodedTextSet;
}
*/

/*
int YoloV8Detection::zbarDecode(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_info_t *info, char *zbarInfo) {
  printf("YoloV8Detection:zbarDecode \n");
  CVI_S32 s32Ret = CVI_SUCCESS;

  cvtdl_image_t tmp_image;
  memset(&tmp_image, 0, sizeof(cvtdl_image_t));
  bool cvtRGB888 = true;
  crop_image(srcFrame, &tmp_image, &info->bbox, cvtRGB888);

  uint32_t h = tmp_image.height, w = tmp_image.width, s = tmp_image.stride[0];
  uint8_t *p = tmp_image.pix[0];
  cv::Mat image(h, w, CV_8UC3, p, s);
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, cv::COLOR_RGB2GRAY);

  std::set<std::string> qr_code_set = decode(image_gray);

  // 用来存储拼接后的字符串
  std::string combinedInfo;

    // 遍历集合并拼接字符串
  for (const auto& info : qr_code_set) {
      combinedInfo += info + "; ";  // 用分号和空格分隔每个字符串
  }

  std::strcpy(zbarInfo, combinedInfo.c_str());

  //清空

  return CVI_TDL_SUCCESS;
}
*/

// the bbox featuremap shape is b x 4*regmax x h x w
void YoloV8Detection::decode_bbox_feature_map(int stride, int anchor_idx,
                                              std::vector<float> &decode_box) {
  std::string box_name;
  if (bbox_out_names.count(stride)) {
    box_name = bbox_out_names[stride];
  } else if (bbox_class_out_names.count(stride)) {
    box_name = bbox_class_out_names[stride];
  }
  TensorInfo boxinfo = getOutputTensorInfo(box_name);

  int num_per_pixel = boxinfo.tensor_size / boxinfo.tensor_elem;
  int8_t *p_box_int8 = static_cast<int8_t *>(boxinfo.raw_pointer);
  float *p_box_float = static_cast<float *>(boxinfo.raw_pointer);
  int num_channel = boxinfo.shape.dim[1];
  int num_anchor = boxinfo.shape.dim[2] * boxinfo.shape.dim[3];
  int box_val_num = 4;
  int reg_max = 16;
  if (m_box_channel_ != box_val_num * reg_max) {
    LOGE("box channel size not ok,got:%d\n", num_channel);
  }

  int32_t feat_w = boxinfo.shape.dim[3];

  int anchor_y = anchor_idx / feat_w;
  int anchor_x = anchor_idx % feat_w;

  // LOGI("box numchannel:%d,numperpixel:%d,featw:%d,feath:%d,anchory:%d,anchorx:%d,numanchor:%d\n",
  //      num_channel, num_per_pixel, feat_w, feat_h, anchor_y, anchor_x, num_anchor);

  float grid_y = anchor_y + 0.5;
  float grid_x = anchor_x + 0.5;

  std::vector<float> grid_logits;  // 4x16
  if (num_per_pixel == 1) {
    for (int c = 0; c < m_box_channel_; c++) {
      grid_logits.push_back(p_box_int8[c * num_anchor + anchor_idx] * boxinfo.qscale);
    }
  } else {
    for (int c = 0; c < m_box_channel_; c++) {
      grid_logits.push_back(p_box_float[c * num_anchor + anchor_idx]);
    }
  }

  // compute softmax and accumulate val per 16
  std::vector<float> box_vals;
  for (int i = 0; i < box_val_num; i++) {
    float sum_softmax = 0;
    float sum_val = 0;
    for (int j = 0; j < reg_max; j++) {
      float expv = exp(grid_logits[i * reg_max + j]);
      sum_softmax += expv;
      sum_val += expv * j;
    }
    sum_softmax = sum_val / sum_softmax;
    box_vals.push_back(sum_softmax);
  }

  std::vector<float> box = {(grid_x - box_vals[0]) * stride, (grid_y - box_vals[1]) * stride,
                            (grid_x + box_vals[2]) * stride, (grid_y + box_vals[3]) * stride};
  decode_box = box;
}

void YoloV8Detection::outputParser(const int image_width, const int image_height,
                                   const int frame_width, const int frame_height,
                                   cvtdl_object_t *obj_meta) {
  Detections vec_obj;
  CVI_SHAPE shape = getInputShape(0);
  int nn_width = shape.dim[3];
  int nn_height = shape.dim[2];
  float inverse_th = std::log(m_model_threshold / (1 - m_model_threshold));

  for (size_t i = 0; i < strides.size(); i++) {
    int stride = strides[i];
    std::string cls_name;
    int cls_offset = 0;
    if (class_out_names.count(stride)) {
      cls_name = class_out_names[stride];
    } else if (bbox_class_out_names.count(stride)) {
      cls_name = bbox_class_out_names[stride];
      cls_offset = m_box_channel_;
    }
    TensorInfo classinfo = getOutputTensorInfo(cls_name);

    int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
    int8_t *p_cls_int8 = static_cast<int8_t *>(classinfo.raw_pointer);
    float *p_cls_float = static_cast<float *>(classinfo.raw_pointer);

    int num_cls = alg_param_.cls;
    int num_anchor = classinfo.shape.dim[2] * classinfo.shape.dim[3];
    // LOGI("stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d\n", stride,
    // classinfo.shape.dim[3],
    //     classinfo.shape.dim[2], num_per_pixel, num_cls);
    float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;
    for (int j = 0; j < num_anchor; j++) {
      int max_logit_c = -1;
      float max_logit = -1000;
      if (num_per_pixel == 1) {
        parse_cls_info<int8_t>(p_cls_int8, num_anchor, num_cls, j, cls_offset, cls_qscale,
                               &max_logit, &max_logit_c);
      } else {
        parse_cls_info<float>(p_cls_float, num_anchor, num_cls, j, cls_offset, cls_qscale,
                              &max_logit, &max_logit_c);
      }
      if (max_logit < inverse_th) {
        continue;
      }
      float score = 1 / (1 + exp(-max_logit));
      std::vector<float> box;
      decode_bbox_feature_map(stride, j, box);
      PtrDectRect det = std::make_shared<object_detect_rect_t>();
      det->score = score;
      det->x1 = box[0];
      det->y1 = box[1];
      det->x2 = box[2];
      det->y2 = box[3];
      det->label = max_logit_c;
      clip_bbox(nn_width, nn_height, det);
      float box_width = det->x2 - det->x1;
      float box_height = det->y2 - det->y1;
      if (box_width > 1 && box_height > 1) {
        vec_obj.push_back(det);
      }
    }
  }
  postProcess(vec_obj, frame_width, frame_height, obj_meta, true);
}

void YoloV8Detection::parseDecodeBranch(const int image_width, const int image_height,
                                        const int frame_width, const int frame_height,
                                        cvtdl_object_t *obj_meta) {
  int stride = strides[0];
  TensorInfo oinfo_box = getOutputTensorInfo(bbox_out_names[stride]);
  TensorInfo oinfo_cls = getOutputTensorInfo(class_out_names[stride]);

  int num_per_pixel_cls = oinfo_cls.tensor_size / oinfo_cls.tensor_elem;
  int8_t *p_cls_int8 = static_cast<int8_t *>(oinfo_cls.raw_pointer);
  float *p_cls_float = static_cast<float *>(oinfo_cls.raw_pointer);

  int num_per_pixel_box = oinfo_box.tensor_size / oinfo_box.tensor_elem;
  int8_t *p_box_int8 = static_cast<int8_t *>(oinfo_box.raw_pointer);
  float *p_box_float = static_cast<float *>(oinfo_box.raw_pointer);

  int num_cls = alg_param_.cls;
  float cls_qscale = num_per_pixel_cls == 1 ? oinfo_cls.qscale : 1;
  float box_qscale = num_per_pixel_box == 1 ? oinfo_box.qscale : 1;
  int cls_offset = 0;

  Detections vec_obj;

  CVI_SHAPE shape = getInputShape(0);
  // y = 1/(1+exp(-x)) ==>
  float inverse_th = std::log(m_model_threshold / (1 - m_model_threshold));
  int num_anchor = oinfo_cls.shape.dim[2];

  LOGI("parseDecodeBranch box_pixel:%d,cls_pixel:%d,numanchor:%d\n", num_per_pixel_cls,
       num_per_pixel_box, num_anchor);

  float x, y, w, h;
  for (int i = 0; i < num_anchor; i++) {
    int max_logit_c = -1;
    float max_logit = -1000;
    if (num_per_pixel_cls == 1) {
      parse_cls_info<int8_t>(p_cls_int8, num_anchor, num_cls, i, cls_offset, cls_qscale, &max_logit,
                             &max_logit_c);
    } else {
      parse_cls_info<float>(p_cls_float, num_anchor, num_cls, i, cls_offset, cls_qscale, &max_logit,
                            &max_logit_c);
    }
    if (max_logit < inverse_th) {
      continue;
    }
    float score = 1 / (1 + exp(-max_logit));
    if (num_per_pixel_box == 1) {
      x = p_box_int8[0 * num_anchor + i] * box_qscale;
      y = p_box_int8[1 * num_anchor + i] * box_qscale;
      w = p_box_int8[2 * num_anchor + i] * box_qscale;
      h = p_box_int8[3 * num_anchor + i] * box_qscale;
    } else {
      x = p_box_float[0 * num_anchor + i];
      y = p_box_float[1 * num_anchor + i];
      w = p_box_float[2 * num_anchor + i];
      h = p_box_float[3 * num_anchor + i];
    }

    int x1 = int((x - 0.5 * w));
    int y1 = int((y - 0.5 * h));

    int x2 = int((x + 0.5 * w));
    int y2 = int((y + 0.5 * h));

    PtrDectRect det = std::make_shared<object_detect_rect_t>();
    det->score = score;
    det->x1 = x1;
    det->y1 = y1;
    det->x2 = x2;
    det->y2 = y2;
    det->label = 0;
    clip_bbox(shape.dim[3], shape.dim[2], det);
    float box_width = det->x2 - det->x1;
    float box_height = det->y2 - det->y1;
    if (box_width > 1 && box_height > 1) {
      vec_obj.push_back(det);
    }
  }
  postProcess(vec_obj, frame_width, frame_height, obj_meta, true);
}
void YoloV8Detection::postProcess(Detections &dets, int frame_width, int frame_height,
                                  cvtdl_object_t *obj_meta, bool rescale) {                           
  Detections final_dets = nms_multi_class(dets, m_model_nms_threshold); 

  CVI_SHAPE shape = getInputShape(0);
  convert_det_struct(final_dets, obj_meta, shape.dim[2], shape.dim[3]);

  if (!hasSkippedVpssPreprocess()) {
    for (uint32_t i = 0; i < obj_meta->size; ++i) {
      if(rescale){
        obj_meta->info[i].bbox = box_rescale(
          frame_width, 
          frame_height, 
          obj_meta->width, 
          obj_meta->height,
          obj_meta->info[i].bbox, 
          meta_rescale_type_e::RESCALE_CENTER
        );
      } else {
        obj_meta->info[i].bbox = obj_meta->info[i].bbox;
      }
    }
    obj_meta->width = frame_width;
    obj_meta->height = frame_height;
  }
}

int YoloV8Detection::outputParser_windowsDetect(const int frame_width, const int frame_height, Detections *vec_obj) {
  CVI_SHAPE shape = getInputShape(0);
  int nn_width = shape.dim[3];
  int nn_height = shape.dim[2];

  float inverse_th = std::log(m_model_threshold / (1 - m_model_threshold));

  int count  = 0;
  for (size_t i = 0; i < strides.size(); i++) {
    int stride = strides[i];
    std::string cls_name;
    int cls_offset = 0;
    if (class_out_names.count(stride)) {
      cls_name = class_out_names[stride];
    } else if (bbox_class_out_names.count(stride)) {
      cls_name = bbox_class_out_names[stride];
      cls_offset = m_box_channel_;
    }
    TensorInfo classinfo = getOutputTensorInfo(cls_name);

    int num_per_pixel = classinfo.tensor_size / classinfo.tensor_elem;
    int8_t *p_cls_int8 = static_cast<int8_t *>(classinfo.raw_pointer);
    float *p_cls_float = static_cast<float *>(classinfo.raw_pointer);

    int num_cls = alg_param_.cls;
    int num_anchor = classinfo.shape.dim[2] * classinfo.shape.dim[3];
    // LOGI("stride:%d,featw:%d,feath:%d,numperpixel:%d,numcls:%d\n", stride,
    // classinfo.shape.dim[3],
    //     classinfo.shape.dim[2], num_per_pixel, num_cls);
    float cls_qscale = num_per_pixel == 1 ? classinfo.qscale : 1;
    for (int j = 0; j < num_anchor; j++) {
      int max_logit_c = -1;
      float max_logit = -1000;
      if (num_per_pixel == 1) {
        parse_cls_info<int8_t>(p_cls_int8, num_anchor, num_cls, j, cls_offset, cls_qscale,
                               &max_logit, &max_logit_c);
      } else {
        parse_cls_info<float>(p_cls_float, num_anchor, num_cls, j, cls_offset, cls_qscale,
                              &max_logit, &max_logit_c);
      }
      if (max_logit < inverse_th) {
        continue;
      }
      float score = 1 / (1 + exp(-max_logit));
      std::vector<float> box;
      decode_bbox_feature_map(stride, j, box);
      PtrDectRect det = std::make_shared<object_detect_rect_t>();
      det->score = score;
      det->x1 = box[0];
      det->y1 = box[1];
      det->x2 = box[2];
      det->y2 = box[3];
      det->label = max_logit_c;
      clip_bbox(nn_width, nn_height, det);
      float box_width = det->x2 - det->x1;
      float box_height = det->y2 - det->y1;
      if (box_width > 1 && box_height > 1) {
        vec_obj->push_back(det);
        count++; 
      }
    }
  }
  return count;
}
// namespace cvitdl
}  // namespace cvitdl
