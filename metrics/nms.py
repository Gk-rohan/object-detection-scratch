import torch
from iou import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Non-maximum Suppression (NMS) on bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[0] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    boxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes if chosen_box[0] != box[0] 
            or intersection_over_union(chosen_box[2:], box[2:], box_format=box_format) < iou_threshold
        ]
    
    return boxes_after_nms