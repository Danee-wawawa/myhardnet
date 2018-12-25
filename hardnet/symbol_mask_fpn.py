import mxnet as mx

from rcnn.config import config
from rcnn.PY_OP import fpn_roi_pooling, proposal_fpn, mask_roi, mask_output, box_annotator_ohem

eps = 2e-5
use_global_stats = True
workspace = 512
HHD_C = 256
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]

def ConvFactory(net, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", name=''):
    net = mx.symbol.Convolution(
        data=net, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name=name+'_conv')
    #net = mx.symbol.BatchNorm(data=net, name=name+'_bn')
    if len(act_type)>0:
      net = mx.symbol.Activation(data=net, act_type=act_type, name=name+'_act')
    return net

def ConvDeformable(net, num_filter, num_group=1, act_type='relu',name=''):
  f = num_group*18
  conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = net,
                      num_filter=f, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
  net = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=net, offset=conv_offset,
                      num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=num_group, stride=(1, 1), no_bias=False)
  #net = mx.symbol.BatchNorm(data=net, name=name+'_bn')
  if len(act_type)>0:
    net = mx.symbol.Activation(data=net, act_type=act_type, name=name+'_act')
  return net

def inception_block(net, num_filter, name):
    assert num_filter%4==0
    f = num_filter//4
    tower_conv0_0 = ConvFactory(net, f, (1, 1), name=name+'_tower_conv0_0')
    tower_conv0_1 = ConvFactory(tower_conv0_0, f, (1, 1), name=name+'_tower_conv0_1')
    tower_conv0_2 = ConvDeformable(tower_conv0_1, f, name=name+'_tower_conv0_2')
    tower_conv1_0 = ConvFactory(net, f, (1, 1), name=name+'_tower_conv1_0')
    tower_conv1_1 = ConvFactory(tower_conv1_0, f, (1, 3), pad=(0, 1), name=name+'_tower_conv1_1')
    tower_conv1_2 = ConvFactory(tower_conv1_1, f, (3, 1), pad=(1, 0), name=name+'_tower_conv1_2')
    tower_conv1_3 = ConvDeformable(tower_conv1_2, f, name=name+'_tower_conv1_3')
    tower_conv2_0 = ConvFactory(net, f, (1, 1), name=name+'_tower_conv2_0')
    tower_conv2_1 = ConvFactory(tower_conv2_0, f, (1, 5), pad=(0, 2), name=name+'_tower_conv2_1')
    tower_conv2_2 = ConvFactory(tower_conv2_1, f, (5, 1), pad=(2, 0), name=name+'_tower_conv2_2')
    tower_conv2_3 = ConvDeformable(tower_conv2_2, f, name=name+'_tower_conv2_3')
    tower_mixed = mx.symbol.Concat(*[tower_conv0_2, tower_conv1_3, tower_conv2_3])
    tower_out = ConvFactory(tower_mixed, f, (1, 1), act_type='', name=name+'_tower_out')
    #shortcut = net
    shortcut = ConvFactory(net, f, (1, 1), name=name+'_shortcut')
    scale = 1.0
    net = shortcut+scale*tower_out
    net = mx.symbol.Activation(data=net, act_type='relu')
    return net

def residual_unit(data, num_filter, stride, dim_match, name):
    bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3   = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3  = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_C4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True,
                             name='stage4_unit%s' % i)
    conv_C5 = unit

    conv_feat = [conv_C5, conv_C4, conv_C3, conv_C2]
    return conv_feat

def get_resnet_conv_down(conv_feat):
    # C5 to P5, 1x1 dimension reduction to 256
    P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")

    # P5 2x upsampling + C4 = P4
    P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='nearest', workspace=512, name='P5_upsampling', num_args=1)
    P4_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
    P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
    P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
    P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

    # P4 2x upsampling + C3 = P3
    P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la   = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    # P6 2x subsampling P5
    P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})
    if config.USE_BPA:
      N2 = P2
      conv_fpn_feat['stride4'] = N2
      body = N2
      for n in range(3, 6):
        body_down = mx.symbol.Convolution(data=body, kernel=(3, 3), stride=(2,2), pad=(1, 1), num_filter=256, name="N%d_subsampling"%n)
        body_down = mx.sym.Activation(data=body_down, act_type='relu', name='N%d_subsampling_relu'%n)
        stridename = "stride%d"%(2**n)
        pbody = conv_fpn_feat[stridename]
        body = mx.sym.ElementWiseSum(*[pbody, body_down], name="N%d_sum"%n)
        body = mx.symbol.Convolution(data=body, kernel=(3, 3), stride=(1,1), pad=(1, 1), num_filter=256, name="N%d_aggregate"%n)
        body = mx.sym.Activation(data=body_down, act_type='relu', name='N%d_aggregate_relu'%n)
        conv_fpn_feat[stridename] = body

    if config.USE_INCEPT:
      for k in conv_fpn_feat:
        v = conv_fpn_feat[k]
        if config.USE_INCEPT==2:
          conv_fpn_feat[k] = inception_block(v, 256, k+"_Incep")
        else:
          conv_fpn_feat[k] = inception_block(v, 256, "INCEPT_%s"%k)

    return conv_fpn_feat


def get_resnet_fpn_rpn(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)

    # shared convolutional layers, top down
    conv_fpn_feat = get_resnet_conv_down(conv_feat)

    # shared parameters for predictions
    rpn_conv_weight      = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias        = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight  = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias    = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias   = mx.symbol.Variable('rpn_conv_bbox_bias')

    rpn_cls_score_list = []
    rpn_bbox_pred_list = []
    for stride in config.RPN_FEAT_STRIDE:
        _feat = conv_fpn_feat['stride%s'%stride]
        if config.USE_INCEPT<2:
          rpn_conv = mx.symbol.Convolution(data=_feat,
                                           kernel=(3, 3), pad=(1, 1),
                                           num_filter=512,
                                           weight=rpn_conv_weight,
                                           bias=rpn_conv_bias)
          rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                          act_type="relu",
                                          name="rpn_relu")
        else:
          rpn_relu = _feat
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1),
                                                  name="rpn_cls_score_reshape_stride%s" % stride)
        rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                  shape=(0, 0, -1),
                                                  name="rpn_bbox_pred_reshape_stride%s" % stride)

        rpn_bbox_pred_list.append(rpn_bbox_pred_reshape)
        rpn_cls_score_list.append(rpn_cls_score_reshape)

    # concat output of each level
    rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list, dim=2)
    rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list, dim=2)

    # loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_concat,
                                           label=rpn_label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rpn_cls_prob')

    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                           data=(rpn_bbox_pred_concat - rpn_bbox_target))

    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                    grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    rpn_group = [rpn_cls_prob, rpn_bbox_loss]
    group = mx.symbol.Group(rpn_group)
    return group


def get_resnet_fpn_rpn_test(num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)
    conv_fpn_feat = get_resnet_conv_down(conv_feat)

    # # shared parameters for predictions
    rpn_conv_weight      = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias        = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight  = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias    = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias   = mx.symbol.Variable('rpn_conv_bbox_bias')

    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for stride in config.RPN_FEAT_STRIDE:
        _feat = conv_fpn_feat['stride%s'%stride]
        if config.USE_INCEPT<2:
          rpn_conv = mx.symbol.Convolution(data=_feat,
                                           kernel=(3, 3), pad=(1, 1),
                                           num_filter=512,
                                           weight=rpn_conv_weight,
                                           bias=rpn_conv_bias)
          rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                          act_type="relu",
                                          name="rpn_relu")
        else:
          rpn_relu = _feat
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s" % stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # ROI Proposal
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_reshape')

        rpn_cls_prob_dict.update({'cls_prob_stride%s' % stride: rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'bbox_pred_stride%s' % stride: rpn_bbox_pred})
    args_dict = dict(rpn_cls_prob_dict.items()+rpn_bbox_pred_dict.items())
    aux_dict = {'im_info':im_info,'name':'rois',
                'op_type':'proposal_fpn','output_score':True,
                'feat_stride':config.RPN_FEAT_STRIDE,'scales':tuple(config.ANCHOR_SCALES),
                'ratios':tuple(config.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n':config.TEST.PROPOSAL_PRE_NMS_TOP_N,
                'rpn_post_nms_top_n':config.TEST.PROPOSAL_POST_NMS_TOP_N,
                'rpn_min_size':config.TEST.RPN_MIN_SIZE,
                'threshold':config.TEST.RPN_NMS_THRESH}
    # Proposal
    group = mx.symbol.Custom(**dict(args_dict.items()+aux_dict.items()))

    # rois = group[0]
    # score = group[1]
    return group


def get_resnet_fpn_maskrcnn(num_classes=config.NUM_CLASSES):
    rcnn_feat_stride = config.RCNN_FEAT_STRIDE
    data = mx.symbol.Variable(name="data")
    if not config.USE_AFP:
      rois = dict()
      label = dict()
      bbox_target = dict()
      bbox_weight = dict()
      mask_target = dict()
      mask_weight = dict()
      for s in rcnn_feat_stride:
          rois['rois_stride%s' % s] = mx.symbol.Variable(name='rois_stride%s' % s)
          label['label_stride%s' % s] = mx.symbol.Variable(name='label_stride%s' % s)
          bbox_target['bbox_target_stride%s' % s] = mx.symbol.Variable(name='bbox_target_stride%s' % s)
          bbox_weight['bbox_weight_stride%s' % s] = mx.symbol.Variable(name='bbox_weight_stride%s' % s)
          mask_target['mask_target_stride%s' % s] = mx.symbol.Variable(name='mask_target_stride%s' % s)
          mask_weight['mask_weight_stride%s' % s] = mx.symbol.Variable(name='mask_weight_stride%s' % s)

      # reshape input
      for s in rcnn_feat_stride:
          rois['rois_stride%s' % s] = mx.symbol.Reshape(data=rois['rois_stride%s' % s],
                                                        shape=(-1, 5),
                                                        name='rois_stride%s_reshape' % s)

          label['label_stride%s' % s] = mx.symbol.Reshape(data=label['label_stride%s' % s], shape=(-1,), name='label_stride%s_reshape'%s)
          bbox_target['bbox_target_stride%s' % s] = mx.symbol.Reshape(data=bbox_target['bbox_target_stride%s' % s],
                                                                      shape=(-1, 4 * num_classes),
                                                                      name='bbox_target_stride%s_reshape'%s)
          bbox_weight['bbox_weight_stride%s' % s] = mx.symbol.Reshape(data=bbox_weight['bbox_weight_stride%s' % s],
                                                                      shape=(-1, 4 * num_classes),
                                                                      name='bbox_weight_stride%s_reshape'%s)
          mask_target['mask_target_stride%s' % s] = mx.symbol.Reshape(data=mask_target['mask_target_stride%s' % s],
                                                                      shape=(-1, num_classes, config.ROI_SIZE[0]*2, config.ROI_SIZE[1]*2),
                                                                      name='mask_target_stride%s_reshape'%s)
          mask_weight['mask_weight_stride%s' % s] = mx.symbol.Reshape(data=mask_weight['mask_weight_stride%s' % s],
                                                                      shape=(-1, num_classes, 1, 1),
                                                                      name='mask_weight_stride%s_reshape'%s)

      label_list = []
      bbox_target_list = []
      bbox_weight_list = []
      mask_target_list = []
      mask_weight_list = []
      for s in rcnn_feat_stride:
          label_list.append(label['label_stride%s' % s])
          bbox_target_list.append(bbox_target['bbox_target_stride%s' % s])
          bbox_weight_list.append(bbox_weight['bbox_weight_stride%s' % s])
          mask_target_list.append(mask_target['mask_target_stride%s' % s])
          mask_weight_list.append(mask_weight['mask_weight_stride%s' % s])

      label = mx.symbol.concat(*label_list, dim=0)
      bbox_target = mx.symbol.concat(*bbox_target_list, dim=0)
      bbox_weight = mx.symbol.concat(*bbox_weight_list, dim=0)
      mask_target = mx.symbol.concat(*mask_target_list, dim=0)
      mask_weight = mx.symbol.concat(*mask_weight_list, dim=0)
    else: #use afp
      rois = mx.symbol.Variable(name='rois')
      label = mx.symbol.Variable(name='label')
      bbox_target = mx.symbol.Variable(name='bbox_target')
      bbox_weight = mx.symbol.Variable(name='bbox_weight')
      mask_target = mx.symbol.Variable(name='mask_target')
      mask_weight = mx.symbol.Variable(name='mask_weight')
      rois = mx.symbol.Reshape(data=rois,
                               shape=(-1, 5),
                               name='rois_reshape')

      label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
      bbox_target = mx.symbol.Reshape(data=bbox_target,
                                      shape=(-1, 4 * num_classes),
                                      name='bbox_target_reshape')
      bbox_weight = mx.symbol.Reshape(data=bbox_weight,
                                      shape=(-1, 4 * num_classes),
                                      name='bbox_weight_reshape')
      mask_target = mx.symbol.Reshape(data=mask_target,
                                      shape=(-1, num_classes, config.ROI_SIZE[0]*2, config.ROI_SIZE[1]*2),
                                      name='mask_target_reshape')
      mask_weight = mx.symbol.Reshape(data=mask_weight,
                                      shape=(-1, num_classes, 1, 1),
                                      name='mask_weight_reshape')


    # shared convolutional layers, bottom up
    conv_feat = get_resnet_conv(data)

    # shared convolutional layers, top down
    conv_fpn_feat = get_resnet_conv_down(conv_feat)

    # shared parameters for predictions
    if config.USE_HHD:
      rcnn_fc4_weight = mx.symbol.Variable('rcnn_fc4_weight')
      rcnn_fc4_bias   = mx.symbol.Variable('rcnn_fc4_bias')
      rcnn_fc5_weight = mx.symbol.Variable('rcnn_fc5_weight')
      rcnn_fc5_bias   = mx.symbol.Variable('rcnn_fc5_bias')
    rcnn_fc6_weight     = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias       = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight     = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias       = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight  = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias    = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('rcnn_fc_bbox_weight')
    rcnn_fc_bbox_bias   = mx.symbol.Variable('rcnn_fc_bbox_bias')

    mask_conv_1_weight = mx.symbol.Variable('mask_conv_1_weight')
    mask_conv_1_bias   = mx.symbol.Variable('mask_conv_1_bias')
    mask_conv_2_weight = mx.symbol.Variable('mask_conv_2_weight')
    mask_conv_2_bias   = mx.symbol.Variable('mask_conv_2_bias')
    mask_conv_3_weight = mx.symbol.Variable('mask_conv_3_weight')
    mask_conv_3_bias   = mx.symbol.Variable('mask_conv_3_bias')
    mask_conv_4_weight = mx.symbol.Variable('mask_conv_4_weight')
    mask_conv_4_bias   = mx.symbol.Variable('mask_conv_4_bias')
    mask_deconv_1_weight = mx.symbol.Variable('mask_deconv_1_weight')
    mask_deconv_2_weight = mx.symbol.Variable('mask_deconv_2_weight')
    mask_deconv_2_bias = mx.symbol.Variable('mask_deconv_2_bias')
    if config.USE_FF:
      mask_conv_4_fc_weight = mx.symbol.Variable('mask_conv_4_fc_weight')
      mask_conv_4_fc_bias = mx.symbol.Variable('mask_conv_4_fc_bias')
      mask_conv_5_fc_weight = mx.symbol.Variable('mask_conv_5_fc_weight')
      mask_conv_5_fc_bias = mx.symbol.Variable('mask_conv_5_fc_bias')
      mask_conv_6_fc_weight = mx.symbol.Variable('mask_conv_6_fc_weight')
      mask_conv_6_fc_bias = mx.symbol.Variable('mask_conv_6_fc_bias')

    rcnn_cls_score_list = []
    rcnn_bbox_pred_list = []
    mask_deconv_act_list = []
    if not config.USE_AFP:
      for stride in rcnn_feat_stride:
          if config.ROIALIGN:
              roi_pool = mx.symbol.contrib.ROIAlign_v2(
                  name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois['rois_stride%s' % stride],
                  pooled_size=(config.ROI_SIZE[0], config.ROI_SIZE[1]),
                  spatial_scale=1.0 / stride)
          else:
              roi_pool = mx.symbol.ROIPooling(
                  name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois['rois_stride%s' % stride],
                  pooled_size=(config.ROI_SIZE[0], config.ROI_SIZE[1]),
                  spatial_scale=1.0 / stride)

          # classification with fc layers
          if not config.USE_HHD:
            flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
            fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
            relu6   = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
            drop6   = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
            fc7     = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
            relu7   = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
          else:
            fc4 = mx.symbol.Convolution(data=roi_pool,
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=HHD_C,
                                             weight=rcnn_fc4_weight,
                                             bias=rcnn_fc4_bias, name='rcnn_fc4')
            relu4 = mx.symbol.Activation(data=fc4,
                                            act_type="relu",
                                            name="rcnn_relu4")
            fc5 = mx.symbol.Convolution(data=relu4,
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=HHD_C,
                                             weight=rcnn_fc5_weight,
                                             bias=rcnn_fc5_bias, name='rcnn_fc5')
            relu5 = mx.symbol.Activation(data=fc5,
                                            act_type="relu",
                                            name="rcnn_relu5")
            fc6 = mx.symbol.Convolution(data=relu5,
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=HHD_C,
                                             weight=rcnn_fc6_weight,
                                             bias=rcnn_fc6_bias, name='rcnn_fc6')
            relu6 = mx.symbol.Activation(data=fc6,
                                            act_type="relu",
                                            name="rcnn_relu6")
            fc7 = mx.symbol.Convolution(data=relu6,
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=HHD_C,
                                             weight=rcnn_fc7_weight,
                                             bias=rcnn_fc7_bias, name='rcnn_fc7')
            relu7 = mx.symbol.Activation(data=fc7,
                                            act_type="relu",
                                            name="rcnn_relu7")

          # classification
          cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight, bias=rcnn_fc_cls_bias,
                                               num_hidden=num_classes)
          # bounding box regression
          bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight, bias=rcnn_fc_bbox_bias,
                                               num_hidden=num_classes * 4)
          rcnn_cls_score_list.append(cls_score)
          rcnn_bbox_pred_list.append(bbox_pred)

          # MASK
          mask_conv_1 = mx.symbol.Convolution(
              data=roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_1_weight, bias=mask_conv_1_bias,
              name="mask_conv_1")
          mask_relu_1 = mx.symbol.Activation(data=mask_conv_1, act_type="relu", name="mask_relu_1")
          mask_conv_2 = mx.symbol.Convolution(
              data=mask_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_2_weight, bias=mask_conv_2_bias,
              name="mask_conv_2")
          mask_relu_2 = mx.symbol.Activation(data=mask_conv_2, act_type="relu", name="mask_relu_2")
          mask_conv_3 = mx.symbol.Convolution(
              data=mask_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_3_weight, bias=mask_conv_3_bias,
              name="mask_conv_3")
          mask_relu_3 = mx.symbol.Activation(data=mask_conv_3, act_type="relu", name="mask_relu_3")
          mask_conv_4 = mx.symbol.Convolution(
              data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_weight, bias=mask_conv_4_bias,
              name="mask_conv_4")
          mask_relu_4 = mx.symbol.Activation(data=mask_conv_4, act_type="relu", name="mask_relu_4")
          mask_deconv_1 = mx.symbol.Deconvolution(data=mask_relu_4, kernel=(4, 4), stride=(2, 2), num_filter=256, pad=(1, 1),
                                                  workspace=512, weight=mask_deconv_1_weight, name="mask_deconv1")
          mask_deconv_2 = mx.symbol.Convolution(data=mask_deconv_1, kernel=(1, 1), num_filter=num_classes,
                                                workspace=512, weight=mask_deconv_2_weight, bias=mask_deconv_2_bias, name="mask_deconv2")
          if config.USE_FF:
            mask_conv_4_fc = mx.symbol.Convolution(
                data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_fc_weight,
                bias=mask_conv_4_fc_bias,
                name="mask_conv_4_fc")
            mask_relu_4_fc = mx.symbol.Activation(data=mask_conv_4_fc, act_type="relu", name="mask_relu_4_fc")
            mask_conv_5_fc = mx.symbol.Convolution(
                data=mask_relu_4_fc, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=512, weight=mask_conv_5_fc_weight,
                bias=mask_conv_5_fc_bias,
                name="mask_conv_5_fc")
            mask_relu_5_fc = mx.symbol.Activation(data=mask_conv_5_fc, act_type="relu", name="mask_relu_5_fc")
            mask_conv_6_fc = mx.symbol.FullyConnected(data=mask_relu_5_fc, weight=mask_conv_6_fc_weight,
                                                      bias=mask_conv_6_fc_bias, num_hidden=config.ROI_SIZE[0]*config.ROI_SIZE[1]*4) #B, 784, 1, 1
            mask_relu_6_fc = mx.symbol.Activation(data=mask_conv_6_fc, act_type="relu", name="mask_relu_6_fc")
            #need relu?
            mask_relu_6_fc_reshape = mx.symbol.Reshape(mask_relu_6_fc, shape=(-1, 1, config.ROI_SIZE[0]*2, config.ROI_SIZE[1]*2))
            mask_deconv_2 = mx.symbol.broadcast_add(mask_deconv_2, mask_relu_6_fc_reshape)
          mask_deconv_act_list.append(mask_deconv_2)

      # concat output of each level
      cls_score_concat = mx.symbol.concat(*rcnn_cls_score_list, dim=0)  # [num_rois_4level, num_class]
      bbox_pred_concat = mx.symbol.concat(*rcnn_bbox_pred_list, dim=0)  # [num_rois_4level, num_class*4]
      mask_act_concat = mx.symbol.concat(*mask_deconv_act_list, dim=0)

    else: #use afp
      box_feat = None
      mask_feat = None
      for stride in rcnn_feat_stride:
          if config.ROIALIGN:
              roi_pool = mx.symbol.contrib.ROIAlign_v2(
                  name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois,
                  pooled_size=(config.ROI_SIZE[0], config.ROI_SIZE[1]),
                  spatial_scale=1.0 / stride)
          else:
              roi_pool = mx.symbol.ROIPooling(
                  name='roi_pool', data=conv_fpn_feat['stride%s'%stride], rois=rois,
                  pooled_size=(config.ROI_SIZE[0], config.ROI_SIZE[1]),
                  spatial_scale=1.0 / stride)
          if not config.USE_HHD:
            flatten = mx.symbol.Flatten(data=roi_pool, name="flatten")
            fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
            relu6   = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
            drop6   = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
            fuse_box = drop6
          else:
            fc4 = mx.symbol.Convolution(data=roi_pool,
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=HHD_C,
                                             weight=rcnn_fc4_weight,
                                             bias=rcnn_fc4_bias, name='rcnn_fc4')
            relu4 = mx.symbol.Activation(data=fc4,
                                            act_type="relu",
                                            name="rcnn_relu4")
            fuse_box = relu4
          mask_conv_1 = mx.symbol.Convolution(
              data=roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_1_weight, bias=mask_conv_1_bias,
              name="mask_conv_1")
          mask_relu_1 = mx.symbol.Activation(data=mask_conv_1, act_type="relu", name="mask_relu_1")
          fuse_mask = mask_relu_1
          if box_feat is None:
            box_feat = fuse_box
            mask_feat = fuse_mask
          else:
            box_feat = mx.symbol.maximum(box_feat, fuse_box)
            mask_feat = mx.symbol.maximum(mask_feat, fuse_mask)
          #box_feat_list.append(fuse_box)
          #mask_feat_list.append(fuse_mask)
      if not config.USE_HHD:
        fc7     = mx.symbol.FullyConnected(data=box_feat, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
        relu7   = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
      else:
        fc5 = mx.symbol.Convolution(data=box_feat,
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=HHD_C,
                                         weight=rcnn_fc5_weight,
                                         bias=rcnn_fc5_bias, name='rcnn_fc5')
        relu5 = mx.symbol.Activation(data=fc5,
                                        act_type="relu",
                                        name="rcnn_relu5")
        fc6 = mx.symbol.Convolution(data=relu5,
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=HHD_C,
                                         weight=rcnn_fc6_weight,
                                         bias=rcnn_fc6_bias, name='rcnn_fc6')
        relu6 = mx.symbol.Activation(data=fc6,
                                        act_type="relu",
                                        name="rcnn_relu6")
        fc7 = mx.symbol.Convolution(data=relu6,
                                         kernel=(3, 3), pad=(1, 1),
                                         num_filter=HHD_C,
                                         weight=rcnn_fc7_weight,
                                         bias=rcnn_fc7_bias, name='rcnn_fc7')
        relu7 = mx.symbol.Activation(data=fc7,
                                        act_type="relu",
                                        name="rcnn_relu7")

      # classification
      cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight, bias=rcnn_fc_cls_bias,
                                           num_hidden=num_classes)
      # bounding box regression
      bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight, bias=rcnn_fc_bbox_bias,
                                           num_hidden=num_classes * 4)
      cls_score_concat = cls_score
      bbox_pred_concat = bbox_pred

      mask_conv_2 = mx.symbol.Convolution(
          data=mask_feat, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_2_weight, bias=mask_conv_2_bias,
          name="mask_conv_2")
      mask_relu_2 = mx.symbol.Activation(data=mask_conv_2, act_type="relu", name="mask_relu_2")
      mask_conv_3 = mx.symbol.Convolution(
          data=mask_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_3_weight, bias=mask_conv_3_bias,
          name="mask_conv_3")
      mask_relu_3 = mx.symbol.Activation(data=mask_conv_3, act_type="relu", name="mask_relu_3")
      mask_conv_4 = mx.symbol.Convolution(
          data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_weight, bias=mask_conv_4_bias,
          name="mask_conv_4")
      mask_relu_4 = mx.symbol.Activation(data=mask_conv_4, act_type="relu", name="mask_relu_4")
      mask_deconv_1 = mx.symbol.Deconvolution(data=mask_relu_4, kernel=(4, 4), stride=(2, 2), num_filter=256, pad=(1, 1),
                                              workspace=512, weight=mask_deconv_1_weight, name="mask_deconv1")
      mask_deconv_2 = mx.symbol.Convolution(data=mask_deconv_1, kernel=(1, 1), num_filter=num_classes,
                                            workspace=512, weight=mask_deconv_2_weight, bias=mask_deconv_2_bias, name="mask_deconv2")
      if config.USE_FF:
        mask_conv_4_fc = mx.symbol.Convolution(
            data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_fc_weight,
            bias=mask_conv_4_fc_bias,
            name="mask_conv_4_fc")
        mask_relu_4_fc = mx.symbol.Activation(data=mask_conv_4_fc, act_type="relu", name="mask_relu_4_fc")
        mask_conv_5_fc = mx.symbol.Convolution(
            data=mask_relu_4_fc, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=512, weight=mask_conv_5_fc_weight,
            bias=mask_conv_5_fc_bias,
            name="mask_conv_5_fc")
        mask_relu_5_fc = mx.symbol.Activation(data=mask_conv_5_fc, act_type="relu", name="mask_relu_5_fc")
        mask_conv_6_fc = mx.symbol.FullyConnected(data=mask_relu_5_fc, weight=mask_conv_6_fc_weight,
                                                  bias=mask_conv_6_fc_bias, num_hidden=config.ROI_SIZE[0]*config.ROI_SIZE[1]*4) #B, 784, 1, 1
        mask_relu_6_fc = mx.symbol.Activation(data=mask_conv_6_fc, act_type="relu", name="mask_relu_6_fc")
        #need relu?
        mask_relu_6_fc_reshape = mx.symbol.Reshape(mask_relu_6_fc, shape=(-1, 1, config.ROI_SIZE[0]*2, config.ROI_SIZE[1]*2))
        mask_deconv_2 = mx.symbol.broadcast_add(mask_deconv_2, mask_relu_6_fc_reshape)
      #mask_deconv_act_list.append(mask_deconv_2)
      mask_act_concat = mask_deconv_2
      #add OHEM here?
      if config.ENABLE_OHEM:
        #labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
        #                                               num_reg_classes=num_reg_classes, roi_per_img=config.TRAIN.BATCH_ROIS,
        #                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
        #                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
        pass

    # loss
    cls_prob = mx.symbol.SoftmaxOutput(data=cls_score_concat,
                                           label=label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           name='rcnn_cls_prob')
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='rcnn_bbox_loss_', scalar=1.0,
                                                   data=(bbox_pred_concat - bbox_target))

    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
    rcnn_group = [cls_prob, bbox_loss]
    for ind, name, last_shape in zip(range(len(rcnn_group)), ['cls_prob', 'bbox_loss'], [num_classes, num_classes * 4]):
        rcnn_group[ind] = mx.symbol.Reshape(data=rcnn_group[ind], shape=(config.TRAIN.BATCH_IMAGES, -1, last_shape),
                                            name=name + '_reshape')

    mask_prob = mx.symbol.Activation(data=mask_act_concat, act_type='sigmoid', name="mask_prob")
    mask_output = mx.symbol.Custom(mask_prob=mask_prob, mask_target=mask_target, mask_weight=mask_weight,
                                   label=label, name="mask_output", op_type='MaskOutput')
    mask_group = [mask_output]
    # group output
    group = mx.symbol.Group(rcnn_group+mask_group)
    return group



def get_resnet_fpn_mask_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    conv_feat = get_resnet_conv(data)
    conv_fpn_feat = get_resnet_conv_down(conv_feat)

    # # shared parameters for predictions
    rpn_conv_weight      = mx.symbol.Variable('rpn_conv_weight')
    rpn_conv_bias        = mx.symbol.Variable('rpn_conv_bias')
    rpn_conv_cls_weight  = mx.symbol.Variable('rpn_conv_cls_weight')
    rpn_conv_cls_bias    = mx.symbol.Variable('rpn_conv_cls_bias')
    rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
    rpn_conv_bbox_bias   = mx.symbol.Variable('rpn_conv_bbox_bias')

    if config.USE_HHD:
      rcnn_fc4_weight = mx.symbol.Variable('rcnn_fc4_weight')
      rcnn_fc4_bias   = mx.symbol.Variable('rcnn_fc4_bias')
      rcnn_fc5_weight = mx.symbol.Variable('rcnn_fc5_weight')
      rcnn_fc5_bias   = mx.symbol.Variable('rcnn_fc5_bias')
    rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
    rcnn_fc6_bias   = mx.symbol.Variable('rcnn_fc6_bias')
    rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
    rcnn_fc7_bias   = mx.symbol.Variable('rcnn_fc7_bias')
    rcnn_fc_cls_weight  = mx.symbol.Variable('rcnn_fc_cls_weight')
    rcnn_fc_cls_bias    = mx.symbol.Variable('rcnn_fc_cls_bias')
    rcnn_fc_bbox_weight = mx.symbol.Variable('rcnn_fc_bbox_weight')
    rcnn_fc_bbox_bias   = mx.symbol.Variable('rcnn_fc_bbox_bias')

    mask_conv_1_weight = mx.symbol.Variable('mask_conv_1_weight')
    mask_conv_1_bias = mx.symbol.Variable('mask_conv_1_bias')
    mask_conv_2_weight = mx.symbol.Variable('mask_conv_2_weight')
    mask_conv_2_bias = mx.symbol.Variable('mask_conv_2_bias')
    mask_conv_3_weight = mx.symbol.Variable('mask_conv_3_weight')
    mask_conv_3_bias = mx.symbol.Variable('mask_conv_3_bias')
    mask_conv_4_weight = mx.symbol.Variable('mask_conv_4_weight')
    mask_conv_4_bias = mx.symbol.Variable('mask_conv_4_bias')
    mask_deconv_1_weight = mx.symbol.Variable('mask_deconv_1_weight')
    mask_deconv_2_weight = mx.symbol.Variable('mask_deconv_2_weight')
    mask_deconv_2_bias = mx.symbol.Variable('mask_deconv_2_bias')

    rpn_cls_prob_dict = {}
    rpn_bbox_pred_dict = {}
    for stride in config.RPN_FEAT_STRIDE:
        _feat = conv_fpn_feat['stride%s'%stride]
        if config.USE_INCEPT<2:
          rpn_conv = mx.symbol.Convolution(data=_feat,
                                           kernel=(3, 3), pad=(1, 1),
                                           num_filter=512,
                                           weight=rpn_conv_weight,
                                           bias=rpn_conv_bias)
          rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                          act_type="relu",
                                          name="rpn_relu")
        else:
          rpn_relu = _feat
        rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=2 * num_anchors,
                                              weight=rpn_conv_cls_weight,
                                              bias=rpn_conv_cls_bias,
                                              name="rpn_cls_score_stride%s"%stride)
        rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                              kernel=(1, 1), pad=(0, 0),
                                              num_filter=4 * num_anchors,
                                              weight=rpn_conv_bbox_weight,
                                              bias=rpn_conv_bbox_bias,
                                              name="rpn_bbox_pred_stride%s" % stride)

        # ROI Proposal
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="rpn_cls_prob_stride%s" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='rpn_cls_prob_reshape')

        rpn_cls_prob_dict.update({'cls_prob_stride%s'%stride:rpn_cls_prob_reshape})
        rpn_bbox_pred_dict.update({'bbox_pred_stride%s'%stride:rpn_bbox_pred})

    args_dict = dict(rpn_cls_prob_dict.items()+rpn_bbox_pred_dict.items())
    aux_dict = {'im_info':im_info,'name':'rois',
                'op_type':'proposal_fpn','output_score':False,
                'feat_stride':config.RPN_FEAT_STRIDE,'scales':tuple(config.ANCHOR_SCALES),
                'ratios':tuple(config.ANCHOR_RATIOS),
                'rpn_pre_nms_top_n':config.TEST.RPN_PRE_NMS_TOP_N,
                'rpn_post_nms_top_n':config.TEST.RPN_POST_NMS_TOP_N,
                'rpn_min_size':config.TEST.RPN_MIN_SIZE,
                'threshold':config.TEST.RPN_NMS_THRESH}
    # Proposal
    rois = mx.symbol.Custom(**dict(args_dict.items()+aux_dict.items()))

    # FPN roi pooling
    args_dict={}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s'%s: conv_fpn_feat['stride%s'%s]})
    args_dict.update({'rois':rois, 'name':'fpn_roi_pool',
                      'op_type':'fpn_roi_pool',
                      'rcnn_strides':config.RCNN_FEAT_STRIDE,
                      'pool_h':config.ROI_SIZE[0], 'pool_w':config.ROI_SIZE[1]})
    roi_pool_fpn = mx.symbol.Custom(**args_dict)

    # classification with fc layers
    if not config.USE_HHD:
      flatten = mx.symbol.Flatten(data=roi_pool_fpn, name="flatten")
      fc6     = mx.symbol.FullyConnected(data=flatten, num_hidden=1024, weight=rcnn_fc6_weight, bias=rcnn_fc6_bias)
      relu6   = mx.symbol.Activation(data=fc6, act_type="relu", name="rcnn_relu6")
      drop6   = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
      fc7     = mx.symbol.FullyConnected(data=drop6, num_hidden=1024, weight=rcnn_fc7_weight, bias=rcnn_fc7_bias)
      relu7   = mx.symbol.Activation(data=fc7, act_type="relu", name="rcnn_relu7")
    else:
      fc4 = mx.symbol.Convolution(data=roi_pool_fpn,
                                       kernel=(3, 3), pad=(1, 1),
                                       num_filter=HHD_C,
                                       weight=rcnn_fc4_weight,
                                       bias=rcnn_fc4_bias, name='rcnn_fc4')
      relu4 = mx.symbol.Activation(data=fc4,
                                      act_type="relu",
                                      name="rcnn_relu4")
      fc5 = mx.symbol.Convolution(data=relu4,
                                       kernel=(3, 3), pad=(1, 1),
                                       num_filter=HHD_C,
                                       weight=rcnn_fc5_weight,
                                       bias=rcnn_fc5_bias, name='rcnn_fc5')
      relu5 = mx.symbol.Activation(data=fc5,
                                      act_type="relu",
                                      name="rcnn_relu5")
      fc6 = mx.symbol.Convolution(data=relu5,
                                       kernel=(3, 3), pad=(1, 1),
                                       num_filter=HHD_C,
                                       weight=rcnn_fc6_weight,
                                       bias=rcnn_fc6_bias, name='rcnn_fc6')
      relu6 = mx.symbol.Activation(data=fc6,
                                      act_type="relu",
                                      name="rcnn_relu6")
      fc7 = mx.symbol.Convolution(data=relu6,
                                       kernel=(3, 3), pad=(1, 1),
                                       num_filter=HHD_C,
                                       weight=rcnn_fc7_weight,
                                       bias=rcnn_fc7_bias, name='rcnn_fc7')
      relu7 = mx.symbol.Activation(data=fc7,
                                      act_type="relu",
                                      name="rcnn_relu7")

    # classification
    rcnn_cls_score = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_cls_weight,
                                              bias=rcnn_fc_cls_bias, num_hidden=num_classes)
    rcnn_cls_prob  = mx.symbol.SoftmaxActivation(name='rcnn_cls_prob', data=rcnn_cls_score)
    # bounding box regression
    rcnn_bbox_pred = mx.symbol.FullyConnected(data=relu7, weight=rcnn_fc_bbox_weight,
                                              bias=rcnn_fc_bbox_bias, num_hidden=num_classes * 4)

    # reshape output
    rcnn_cls_prob  = mx.symbol.Reshape(data=rcnn_cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes),
                                 name='cls_prob_reshape')
    rcnn_bbox_pred = mx.symbol.Reshape(data=rcnn_bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes),
                                 name='bbox_pred_reshape')

    mask_rois = mx.symbol.Custom(data=data, label=rcnn_cls_prob, rois=rois, bbox_deltas=rcnn_bbox_pred,
                                 op_type='mask_roi',
                                 num_classes=num_classes, name='mask_roi')

    args_dict = {}
    for s in config.RCNN_FEAT_STRIDE:
        args_dict.update({'feat_stride%s' % s: conv_fpn_feat['stride%s' % s]})
    args_dict.update({'rois': mask_rois, 'name': 'fpn_maskroi_pool',
                      'op_type': 'fpn_roi_pool',
                      'rcnn_strides': config.RCNN_FEAT_STRIDE,
                      'pool_h': config.ROI_SIZE[0], 'pool_w': config.ROI_SIZE[1]})
    mask_roi_pool = mx.symbol.Custom(**args_dict)
    # MASK
    mask_conv_1 = mx.symbol.Convolution(
        data=mask_roi_pool, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_1_weight,
        bias=mask_conv_1_bias,
        name="mask_conv_1")
    mask_relu_1 = mx.symbol.Activation(data=mask_conv_1, act_type="relu", name="mask_relu_1")
    mask_conv_2 = mx.symbol.Convolution(
        data=mask_relu_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_2_weight,
        bias=mask_conv_2_bias,
        name="mask_conv_2")
    mask_relu_2 = mx.symbol.Activation(data=mask_conv_2, act_type="relu", name="mask_relu_2")
    mask_conv_3 = mx.symbol.Convolution(
        data=mask_relu_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_3_weight,
        bias=mask_conv_3_bias,
        name="mask_conv_3")
    mask_relu_3 = mx.symbol.Activation(data=mask_conv_3, act_type="relu", name="mask_relu_3")



    mask_conv_4 = mx.symbol.Convolution(
        data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_weight,
        bias=mask_conv_4_bias,
        name="mask_conv_4")
    mask_relu_4 = mx.symbol.Activation(data=mask_conv_4, act_type="relu", name="mask_relu_4")
    mask_deconv_1 = mx.symbol.Deconvolution(data=mask_relu_4, kernel=(4, 4), stride=(2, 2), num_filter=256,
                                            pad=(1, 1),
                                            workspace=512, weight=mask_deconv_1_weight, name="mask_deconv1")
    mask_deconv_2 = mx.symbol.Convolution(data=mask_deconv_1, kernel=(1, 1), num_filter=num_classes,
                                          workspace=512, weight=mask_deconv_2_weight, bias=mask_deconv_2_bias,
                                          name="mask_deconv2")

    if config.USE_FF:
      mask_conv_4_fc_weight = mx.symbol.Variable('mask_conv_4_fc_weight')
      mask_conv_4_fc_bias = mx.symbol.Variable('mask_conv_4_fc_bias')
      mask_conv_5_fc_weight = mx.symbol.Variable('mask_conv_5_fc_weight')
      mask_conv_5_fc_bias = mx.symbol.Variable('mask_conv_5_fc_bias')
      mask_conv_6_fc_weight = mx.symbol.Variable('mask_conv_6_fc_weight')
      mask_conv_6_fc_bias = mx.symbol.Variable('mask_conv_6_fc_bias')
      mask_conv_4_fc = mx.symbol.Convolution(
          data=mask_relu_3, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=512, weight=mask_conv_4_fc_weight,
          bias=mask_conv_4_fc_bias,
          name="mask_conv_4_fc")
      mask_relu_4_fc = mx.symbol.Activation(data=mask_conv_4_fc, act_type="relu", name="mask_relu_4_fc")
      mask_conv_5_fc = mx.symbol.Convolution(
          data=mask_relu_4_fc, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=512, weight=mask_conv_5_fc_weight,
          bias=mask_conv_5_fc_bias,
          name="mask_conv_5_fc")
      mask_relu_5_fc = mx.symbol.Activation(data=mask_conv_5_fc, act_type="relu", name="mask_relu_5_fc")
      mask_conv_6_fc = mx.symbol.FullyConnected(data=mask_relu_5_fc, weight=mask_conv_6_fc_weight,
                                                bias=mask_conv_6_fc_bias, num_hidden=config.ROI_SIZE[0]*config.ROI_SIZE[1]*4) #B, 784, 1, 1
      mask_relu_6_fc = mx.symbol.Activation(data=mask_conv_6_fc, act_type="relu", name="mask_relu_6_fc")
      #need relu?
      mask_relu_6_fc_reshape = mx.symbol.Reshape(mask_relu_6_fc, shape=(-1, 1, config.ROI_SIZE[0]*2, config.ROI_SIZE[1]*2))
      mask_deconv_2 = mx.symbol.broadcast_add(mask_deconv_2, mask_relu_6_fc_reshape)

    # group output
    mask_prob = mx.symbol.Activation(data=mask_deconv_2, act_type='sigmoid', name="mask_prob")
    group = mx.symbol.Group([rois, rcnn_cls_prob, rcnn_bbox_pred, mask_prob])
    return group

