from src.pretrain_data import get_loader
from evaluate.metrics4rec import evaluate_all


def evaluate_traditional():
    test_task_list = {'traditional': ['5-8']  # or '5-5'
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

    msg_5_8_1, res_5_8_1 = evaluate_all(ui_scores, gt, 1)
    msg_5_8_5, res_5_8_5 = evaluate_all(ui_scores, gt, 5)
    msg_5_8_10, res_5_8_10 = evaluate_all(ui_scores, gt, 10)

    test_task_list = {'traditional': ['5-5']  # or '5-8'
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

    msg_5_5_1, res_5_5_1 = evaluate_all(ui_scores, gt, 1)
    msg_5_5_5, res_5_5_5 = evaluate_all(ui_scores, gt, 5)
    msg_5_5_10, res_5_5_10 = evaluate_all(ui_scores, gt, 10)

    print("evaluate_traditional")
    print("traditional 5_8")
    print("msg_5_8_1:{} res_5_8_1:{}".format(msg_5_8_1, res_5_8_1))
    print("msg_5_8_5:{} res_5_8_5:{}".format(msg_5_8_5, res_5_8_5))
    print("msg_5_8_10:{} res_5_8_10:{}".format(msg_5_8_10, res_5_8_10))
    print("traditional 5_5")
    print("msg_5_5_1:{} res_5_5_1:{}".format(msg_5_5_1, res_5_5_1))
    print("msg_5_5_5:{} res_5_5_5:{}".format(msg_5_5_5, res_5_5_5))
    print("msg_5_5_10:{} res_5_5_10:{}".format(msg_5_5_10, res_5_5_10))
