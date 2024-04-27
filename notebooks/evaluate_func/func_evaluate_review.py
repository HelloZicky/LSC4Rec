from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

test_task_list = {'review': ['4-4']  # or '4-2'
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

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i > 50:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)

predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))

test_task_list = {'review': ['4-2']  # or '4-4'
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

gt_ratings = []
pred_ratings = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i > 50:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        gt_ratings.extend(batch['target_text'])
        pred_ratings.extend(results)

predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
print('RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
print('MAE {:7.4f}'.format(MAE))

test_task_list = {'review': ['4-1']
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

tokens_predict = []
tokens_test = []
for i, batch in tqdm(enumerate(zeroshot_test_loader)):
    if i > 50:
        break
    with torch.no_grad():
        results = model.generate_step(batch)
        tokens_predict.extend(results)
        tokens_test.extend(batch['target_text'])

new_tokens_predict = [l.split() for l in tokens_predict]
new_tokens_test = [ll.split() for ll in tokens_test]
BLEU2 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=2, smooth=False)
ROUGE = rouge_score(tokens_test, tokens_predict)

print('BLEU-2 {:7.4f}'.format(BLEU2))
for (k, v) in ROUGE.items():
    print('{} {:7.4f}'.format(k, v))