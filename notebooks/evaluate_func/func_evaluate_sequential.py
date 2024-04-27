# from torch.utils.data import DataLoader, Dataset, Sampler
import sys
sys.path.append("./")
# sys.path.append("../")
# sys.path.append("../../")
# sys.path.append("../../../")
from src.pretrain_data import get_loader
from tqdm import tqdm
import torch


def evaluate_sequential(args, model):
    test_task_list = {'sequential': ['2-13']  # or '2-3'
                      }
    test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    print(args.test)
    zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test,
        mode='test',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )
    print(len(zeroshot_test_loader))

    all_info = []
    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            beam_outputs = model.generate(
                batch['input_ids'].to('cuda'),
                max_length=50,
                num_beams=20,
                no_repeat_ngram_size=0,
                num_return_sequences=20,
                early_stopping=True
            )
            generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                new_info = {}
                new_info['target_item'] = item[1]
                new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
                all_info.append(new_info)

    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info['target_item'])]
        pred_dict = {}
        for j in range(len(info['gen_item_list'])):
            try:
                pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            except:
                pass
        ui_scores[i] = pred_dict

    msg_2_13_5, res_2_13_5 = evaluate_all(ui_scores, gt, 5)
    msg_2_13_10, res_2_13_10 = evaluate_all(ui_scores, gt, 10)

    test_task_list = {'sequential': ['2-3']  # or '2-13'
                      }
    test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

    zeroshot_test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test,
        mode='test',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )
    print(len(zeroshot_test_loader))

    all_info = []
    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            beam_outputs = model.generate(
                batch['input_ids'].to('cuda'),
                max_length=50,
                num_beams=20,
                no_repeat_ngram_size=0,
                num_return_sequences=20,
                early_stopping=True
            )
            generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                new_info = {}
                new_info['target_item'] = item[1]
                new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
                all_info.append(new_info)

    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info['target_item'])]
        pred_dict = {}
        for j in range(len(info['gen_item_list'])):
            try:
                pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            except:
                pass
        ui_scores[i] = pred_dict

    msg_2_3_5, res_2_3_5 = evaluate_all(ui_scores, gt, 5)
    msg_2_3_10, res_2_3_10 = evaluate_all(ui_scores, gt, 10)

    print("evaluate_traditional")
    print("traditional 2_13")
    print("msg_2_13_5:{} res_2_13_5:{}".format(msg_2_13_5, res_2_13_5))
    print("msg_2_13_10:{} res_2_13_10:{}".format(msg_2_13_10, res_2_13_10))
    print("traditional 2_3")
    print("msg_2_3_5:{} res_2_3_5:{}".format(msg_2_3_5, res_2_3_5))
    print("msg_2_3_10:{} res_2_3_10:{}".format(msg_2_3_10, res_2_3_10))
