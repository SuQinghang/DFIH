import torch
import time
from loguru import logger   
import torch.nn as nn

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code

def test(model, query_dataloader, retrieval_dataloader, code_length, topk, device):
    load_start = time.time()
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    load_end = time.time()
    
    cal_start = time.time()
    mAP = mean_average_precision(
        query_code.to(device),
        retrieval_code.to(device),
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_dataloader.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    cal_end = time.time()
    logger.debug('[load_time:{:.4f}][cal_time:{:.4f}]'.format(load_end-load_start, cal_end-cal_start))
    return mAP, query_code,retrieval_code

def test_inc(
    model,
    ori_query_dataloader, 
    inc_query_dataloader, 
    inc_retrieval_dataloader, 
    ori_retrieval_code, 
    ori_retrieval_target,
    code_length, 
    topk,
    device,
    ismultilabel=False,
    ori_labels=None,
    inc_labels=None,
    ):
    ori_query_code = generate_code(model, ori_query_dataloader, code_length, device).to(device)
    ori_query_target = ori_query_dataloader.dataset.get_onehot_targets().to(device)
    inc_query_code = generate_code(model, inc_query_dataloader, code_length, device).to(device)
    inc_query_target = inc_query_dataloader.dataset.get_onehot_targets().to(device)
    inc_retrieval_code = generate_code(model, inc_retrieval_dataloader, code_length, device).to(device)
    inc_retrieval_target = inc_retrieval_dataloader.dataset.get_onehot_targets().to(device)
    overall_query_code = torch.vstack((inc_query_code, ori_query_code))
    overall_query_target = torch.vstack((inc_query_target, ori_query_target))
    overall_retrieval_code = torch.vstack((inc_retrieval_code, ori_retrieval_code.to(device)))
    overall_retrieval_target = torch.vstack((inc_retrieval_target, ori_retrieval_target.to(device)))
    if ismultilabel:
        pass
    else:
        inc_map = mean_average_precision(
            inc_query_code.to(device),
            overall_retrieval_code.to(device),
            inc_query_target.to(device),
            overall_retrieval_target.to(device),
            device, 
            topk,
        )
        ori_map = mean_average_precision(
            ori_query_code.to(device),
            overall_retrieval_code.to(device),
            ori_query_target.to(device),
            overall_retrieval_target.to(device),
            device,
            topk,
        )
        ove_map = mean_average_precision(
            overall_query_code.to(device),
            overall_retrieval_code.to(device),
            overall_query_target.to(device),
            overall_retrieval_target.to(device),
            device,
            topk,
        )
    return inc_map, ori_map, ove_map, overall_query_code, overall_query_target, overall_retrieval_code, overall_retrieval_target

