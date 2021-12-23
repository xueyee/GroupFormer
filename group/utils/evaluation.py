import numpy as np
import torch
import collections

def groupMerge(groupList):
    group_to_idx = {}
    graph = collections.defaultdict(set)
    for acc in groupList:
        idx = acc[0]
        for group in acc[1:]:
            graph[acc[1]].add(group)
            graph[group].add(acc[1])
            group_to_idx[group] = idx
    seen = set()
    newGroupList = []
    for group in graph:
        if group not in seen:
            seen.add(group)
            stack = [group]
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nei in graph[node]:
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
            newGroupList.append([group_to_idx[group]] + sorted(component))
    return newGroupList


def judge(gt, det):
    pos = 0
    maxN = max(len(gt), len(det))
    for gtBox in gt:
        if gtBox in det:
            pos += 1
    if pos*1.0/maxN > 0.5:
        return 1
    return 0
def eval_group(boxList, gtGroupList, pred):
    tp, fp, fn, cnt = 0, 0, 0, 0
    boxNumpy = np.array(boxList).squeeze()
    groupTwoList = boxNumpy[pred].tolist()
    cnt = 0
    newBoxList = []

    for boxes in groupTwoList:
        cnt += 1
        newBox = [str(cnt)]
        for box in boxes:
            newBox.append("_".join([str(item) for item in box]))
        newBoxList.append(newBox)

    outGroupList = groupMerge(newBoxList)

    detGroupList = []
    for group in outGroupList:
        tmpGroupList = []
        for box in group[1:]:
            boxStr = box.split('_')
            tmpGroupList.append([int(item) for item in boxStr])
        detGroupList.append(tmpGroupList)
    newGtGroupList = []
    for group in gtGroupList:
        new_group = []
        for box in group:
            new_box = [tmp.item() for tmp in box]
            new_group.append(new_box)
        newGtGroupList.append(new_group)
    #gtList = np.array(gtGroupList).tolist()
    #print(len(gtList))
    gtNum = len(newGtGroupList)
    detNum = len(detGroupList)
    for det in detGroupList:
        for gt in newGtGroupList:
            correct = judge(gt, det)
            if correct:
                tp += 1
                #newGtGroupList.remove(gt)
    fn = torch.tensor(max(gtNum - tp, 0)).cuda()
    fp = torch.tensor(max(detNum - tp, 0)).cuda()
    cnt = torch.tensor(len(detGroupList)).cuda()
    return [torch.tensor(tp).cuda(), fp, fn, cnt]

