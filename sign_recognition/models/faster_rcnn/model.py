import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights


def build_model(number_of_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_of_classes)
    return model


def convert_to_fastercnn_format(bboxes, images, labels):
    targets = [{"boxes": bboxes[i], "labels": labels[i]} for i in range(len(images))]
    return targets
